#!/bin/bash
#SBATCH --job-name=styleptr
#SBATCH --output=logsinfer/st_%j.txt  # output file
#SBATCH -e logsinfer/st_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --gres=gpu:3
#SBATCH --mem=42G
#
#SBATCH --ntasks=1

modelName=simpleModel

declare -A styNo=(
    [STYLED]=2   # Checkpoint for best performing model in training
    [ethnic]=2
)

testName=test

#for sty in STYLED ethnic gender Country edu TOD
#do
#    echo "Running Inference For $sty"
#    CUDA_VISIBLE_DEVICES=0 python -u main.py -mode=inference -method=beam -emb_size=300 -hidden_size=384 -modelName=tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt -problem=$sty | tee logsinfer/Inference_${modelName}_${sty}_${styNo[$sty]}
#    echo "Computing Metrics For $sty"
#    python -u computeNLGEvalMetrics.py tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt.test.beam.output data/${testName}.${sty}.tgt | tee logsinfer/NLGEval_${modelName}_${sty}_${styNo[$sty]}
#    echo "Done for $sty"
#done

modelName=simpleModelGlove2BothPretrainedPointer10_6

for sty in STYLED #Politics ethnic gender Country edu TOD
do
    echo "Running Inference For $sty"
    CUDA_VISIBLE_DEVICES=1 python -u main.py -mode=inference -method=beam -emb_size=300 -hidden_size=384 -modelName=tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt -problem=$sty | tee logsinfer/Inference_${modelName}_${sty}_${styNo[$sty]}
    echo "Computing Metrics For $sty"
    python -u computeNLGEvalMetrics.py tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt.test.beam.output data/${testName}.${sty}.tgt | tee logsinfer/NLGEval_${modelName}_${sty}_${styNo[$sty]}
    echo "Done for $sty"
done
