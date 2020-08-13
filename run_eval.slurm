#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH -c 8
#SBATCH --constraint=gpu32

module load anaconda/anaconda3
source activate pytorch-gpu

python3 inference.py
# python3 evalPixelLevelSemanticLabeling.py
python3 evalInstanceLevelSemanticLabeling.py

