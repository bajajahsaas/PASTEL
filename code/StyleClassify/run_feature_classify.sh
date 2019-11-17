#!/bin/bash
#SBATCH --job-name=stylecls
#SBATCH --output=logs/hh_%j.txt  # output file
#SBATCH -e logs/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:3
#SBATCH --mem=42G
#
#SBATCH --ntasks=1


export CUDA_VISIBLE_DEVICES=0,1,2

PROJ='PASTEL'
DATADIR=../../data/
MODELDIR=../../model/
W2VDIR=../../data/word2vec/glove.840B.300d.txt
MAXFEATURE=70000

#LTYPES=('controlled' 'combined')
#LEVELS=('sentences' 'stories')

LTYPES=('controlled')
LEVELS=('sentences')
#EMD='w2v'
EMB='bert'

for LEVEL in "${LEVELS[@]}"
do
    for LTYPE in "${LTYPES[@]}"
    do
        echo "=============================================="
        echo "Extracting features..." $PROJ $DATADIR $MODELDIR $MAXFEATURE $LEVEL $LTYPE $EMB
        echo "=============================================="
        python -u feature_extract.py \
            $PROJ $DATADIR $MODELDIR $W2VDIR \
            $MAXFEATURE $LEVEL $LTYPE $EMB

        ABLATION=True #False #True
        FCHOOSES=(False) #'deep' 'lexical' 'syntax' False)

        STYLES=('gender' 'age' 'education' 'politics')

        for FCHOOSE in "${FCHOOSES[@]}"
        do
            for STYLE in "${STYLES[@]}"
            do
                echo "=============================================="
                echo "Classifying..." $PROJ $DATADIR $MODELDIR $MAXFEATURE $STYLE $LTYPE $LEVEL $ABLATION $FCHOOSE $FCHOOSE
                echo "=============================================="
                python classify.py \
                    $PROJ $DATADIR $MODELDIR $MAXFEATURE \
                    $STYLE $LEVEL $LTYPE $ABLATION $FCHOOSE
            done
        done
    done
done

exit