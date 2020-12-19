#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH -c 8

module load anaconda/anaconda3
source activate pytorch-gpu

python3 dump_preds.py

