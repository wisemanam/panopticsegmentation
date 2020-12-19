#!/bin/bash
#SBATCH --nodes=1

#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH -c 12
#SBATCH --constraint=gpu32

module load anaconda/anaconda3
source activate pytorch-gpu

python3 train.py

