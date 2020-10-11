import torch
from torch import nn
import math
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        stride: stride of convolution
        dim: size of pose vector

    Shape:
        input:  (*, A, h, w)
        output: poses and activations - (*, B*dim, h', w'), (*, B, h', w')
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*dim + B*dim
    """
    def __init__(self, A, B, K=(3, 3), stride=(1, 1), dim=16):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*dim, kernel_size=K, stride=stride, bias=True)
        self.pose.weight.data.normal_(0.0, 0.1)
        self.a = nn.Conv2d(in_channels=A, out_channels=B, kernel_size=K, stride=stride, bias=True)
        self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()

        self.noise_scale = 4.0

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)

        if self.training:
            a += ((torch.rand(*a.shape) - 0.5) * self.noise_scale).cuda()

        a = self.sigmoid(a)

        return p, a  # Shape (*, B*dim, h', w'), (*, B, h', w')


class CapsulePooling(nn.Module):
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(CapsulePooling, self).__init__()

        self.pooling = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, capsules):
        # shape (*, h', w', B*(P*P+1))

        capsules = capsules.permute(0, 3, 1, 2).contiguous()

        pooled_capsules = self.pooling(capsules)

        return pooled_capsules.permute(0, 2, 3, 1).contiguous()


class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B, C, K=(3, 3), P=4, stride=(1, 1), iters=3, coor_add=False, w_shared=False, padding=None):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        self._lambda = 1e-3  # could have this as lower value for more stability
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsul/home/bruce/projects/capsulese type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.randn(C,self.psize))
        self.beta_a = nn.Parameter(torch.randn(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K[0]*K[1]*B, C, P, P))

        if padding is None:
            self.padding = (0, 0, 0, 0, 0, 0)
        else:
            assert len(padding) == 2
            padding_h, padding_w = padding
            self.padding = (0, 0, padding_w, padding_w, padding_h, padding_h)

        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        cost_h = cost_h.sum(dim=2)

        cost_h_mean = torch.mean(cost_h, dim=1, keepdim=True)

        cost_h_stdv = torch.sqrt(torch.sum(cost_h - cost_h_mean,dim=1,keepdim=True)**2 / C + eps)

        a_out = self.sigmoid(self._lambda*(self.beta_a - (cost_h_mean -cost_h)/(cost_h_stdv + eps)))

        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq + eps) - torch.log(sigma_sq.sqrt() + eps) - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(eps + a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K=(3, 3), psize=4, stride=(1, 1)):
        b, h, w, c = x.shape
        assert c == B * (psize + 1)

        oh = int((h - K[0] + 1) / stride[0])
        ow = int((w - K[1] + 1) / stride[1])

        idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - K[0] + 1, stride[0])] for k_idx in range(0, K[0])]
        idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - K[1] + 1, stride[1])] for k_idx in range(0, K[1])]

        x = x[:, idxs_h, :, :]
        x = x[:, :, :, idxs_w, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        x = F.pad(x, self.padding)
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()

            p_in=p_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, 1)

            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            # print('conv cap activations',a_out[0].sum().item(),a_out[0].size())
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K[0] and 1 == self.K[1]
            assert 1 == self.stride[0] and 1 == self.stride[1]
            # assert 1 == self.K
            # assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


