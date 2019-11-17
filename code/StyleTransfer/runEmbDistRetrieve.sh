#!/bin/bash
#SBATCH --job-name=styletrans
#SBATCH --output=logs/st_%j.txt  # output file
#SBATCH -e logs/st_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:3
#SBATCH --mem=42G
#
#SBATCH --ntasks=1


export CUDA_VISIBLE_DEVICES=0,1,2

python -u findDeleteLexicon.py STYLED 7
python -u computeNLGEvalMetrics.py data/test.STYLED.src.delete data/test.STYLED.tgt
