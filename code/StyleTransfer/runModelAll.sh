#!/bin/bash
#SBATCH --job-name=styletrans
#SBATCH --output=logsmodel/st_%j.txt  # output file
#SBATCH -e logsmodel/st_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:3
#SBATCH --mem=42G
#
#SBATCH --ntasks=1

#modelName=simpleModelGlove2

#for sty in STYLED ethnic gender Country edu TOD
#do
#    echo "----------------------"
#    echo "Training Model For $sty"
#    CUDA_VISIBLE_DEVICES=2 python -u main.py -mode=train -initGlove2 -modelName=${modelName}_$sty -emb_size=300 -hidden_size=384  -problem=$sty -NUM_EPOCHS=7 | tee logsmodel/${modelName}_$sty
#    echo "Finished Training Model for $sty"
#    echo "----------xxx------------------"
#done

modelName=simpleModelGlove2BothPretrained

for sty in STYLED #ethnic gender Country edu TOD #Politics
do
    echo "----------------------"
    echo "Training Model For $sty"
    CUDA_VISIBLE_DEVICES="1" python -u main.py -mode=train -preTrain -NUM_PRETRAIN_EPOCHS=4 -initGlove2 -initGloveEncode2 -modelName=${modelName}_$sty -emb_size=300 -hidden_size=384  -problem=$sty -NUM_EPOCHS=6 | tee logsmodel/${modelName}_$sty
    echo "Finished Training Model for $sty"
    echo "----------xxx------------------"
done

#300 384 - best so far

#for sty in STYLED-story ethnic-story gender-story Country-story edu-story TOD-story
#do
#    echo "----------------------"
#    echo "Training Model For $sty"
#    CUDA_VISIBLE_DEVICES=2 python -u main.py -mode=train -batch_size=8  -sigmoid -initGlove2 -initGloveEncode2 -modelName=${modelName}_$sty -emb_size=300 -hidden_size=384  -problem=$sty -NUM_EPOCHS=15 | tee logsmodel/${modelName}_$sty
#    echo "Finished Training Model for $sty"
#    echo "----------xxx------------------"
#done


#for sty in ethnic-paired gender-paired Country-paired edu-paired TOD-paired STYLED-paired
#do
#    echo "----------------------"
#    echo "Training Model For $sty"
#    CUDA_VISIBLE_DEVICES=0 python -u main.py -mode=train -initGlove -modelName=simpleModelGlove_$sty -emb_size=300 -hidden_size=384  -problem=$sty | tee logsmodel/simpleModelGlove_$sty
#    echo "Finished Training Model for $sty"
#    echo "----------xxx------------------"
#done
