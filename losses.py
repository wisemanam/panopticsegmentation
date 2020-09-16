import torch
from torch import nn
import torch.functional as F


class MarginLoss(nn.Module):
    def __init__(self, reduction='sum', ignore_index=255):
        super(MarginLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true, margin=0.9):
        b, c, h, w = y_pred.shape

        y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, c)
        y_true = y_true.view(-1, )

        if self.ignore_index is not None:
            ignore_mask = (y_true == self.ignore_index)  # 1 where we ignore, 0 otherwise
            y_true[ignore_mask] = 0
        else:
            ignore_mask = torch.zeros_like(y_true)
        ignore_mask = 1.0 - ignore_mask.float()

        a_t = y_pred[range(b * h * w), y_true.long()].unsqueeze(1)

        margin = torch.pow((margin - (a_t - y_pred)).clamp(0, 100000), 2)

        margin = margin * ignore_mask.unsqueeze(1)

        if self.reduction is None:
            return margin.view(b, h, w, c).sum(-1)
        elif self.reduction == 'sum':
            return margin.sum()
        else:
            return margin.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
