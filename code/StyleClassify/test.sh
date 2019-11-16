#!/bin/bash
#SBATCH --job-name=stylecls
#SBATCH --output=logs/hh_%j.txt  # output file
#SBATCH -e logs/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:3
#
#SBATCH --ntasks=1

# shellcheck disable=SC1068

PROJ='PASTEL'
DATADIR=../../data/
MODELDIR=../../model/
W2VDIR=../../data/word2vec/glove.840B.300d.txt
MAXFEATURE=70000

export CUDA_VISIBLE_DEVICES=0,1,2

python test.py
exit