# Textual Style Classification & Transfer on Multiple Personas

__Poster:__ https://drive.google.com/open?id=1oqbIPAuq1PdARAmyVPtiymH-t8rEK3ie

__Results:__ https://docs.google.com/spreadsheets/d/1FHgbpMKFMhklJ3qoC6bru3B0loak1VwB7A4X5Hk44zE/edit?usp=sharing

Data and baselines taken from ["(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas"](https:/arxiv.org/abs/1909.00098) by Dongyeop Kang, Varun Gangal, and Eduard Hovy, EMNLP 2019

## The PASTEL dataset
PASTEL is a parallelly annotated stylistic language dataset.
The dataset consists of ~41K parallel sentences annotated across different personas.

## Approach
Implemented copy-pointer based Seq2Seq models for the task of multi-persona style transfer. Achieved 18% gain in the BLEU score for stylized sentences while retaining the original meaning of the source text.

Achieved 15-20% improvement in F1 scores using BERT-based finetuning models for style classification task.

#### Setup

Git repository: https://github.com/bajajahsaas/Style-Classification-Transfer. This is the fork of the repository by the original authors of PASTEL dataset paper. We have built on top of the baselines for classification and transfer: https://github.com/dykang/PASTEL
Run setup.sh. This file:
Unzips data into data/data_v2.zip directory. We have run our models on the v2 which is the cleaner version of data and has noisy data filtered out.
To run the classification baseline using GloVe or word2vec embeddings, download [GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) into data/word2vec/glove.840B.300d.txt

#### Style Classification

__Dependencies:__
Python 3.7.3
Unzip code/StyleClassify/tsv_bert_files.zip. These tsv files contain preprocessed data that was used for classification (fine-tuning BERT, RoBERTa models).

__Classification:__

Pretrained BERT and GloVe embeddings:
Run StyleClassify/feature_classify.sh. This file extracts features from feature_extract.py as well as performs classification based on the following arguments:
LTYPES: (type of classification to perform) ‘controlled’,’combined’
LEVELS: (classification on what data) ‘sentences’. We have performed style classification on the sentence data.
EMB: (mentions the type of embedding ‘w2v’ for GloVe and ‘bert’ for BERT embeddings) ‘w2v’, ‘bert’
STYLES: (styles to perform classification on) 'gender', 'age','education', 'politics’

__Finetuned BERT:__
https://colab.research.google.com/drive/1r21Q2inVaAm3r3WL8a3jPcRQJYqU3jqf
Upload the tsv files to classify from StyleClassify/tsv_bert_files/$LTYPE$_$STYLE$.tsv and $LTYPE$_$STYLE$_test.tsv for classification. We have performed combined classification on most contradicting style values of the styles.

__Finetuned RoBERTa:__
https://colab.research.google.com/drive/15QM1N8wnEWlFktTGskZw48DVsiETrUxJ
Upload the tsv files to classify from StyleClassify/tsv_bert_files/$LTYPE$_$STYLE$.tsv and $LTYPE$_$STYLE$_test.tsv for classification. We have performed combined classification on most contradicting style values of the styles.

#### Style Transfer

__Dependencies:__
Python 2.7.14
Nlg - eval : https://github.com/Maluuba/nlg-eval
Download wiki word vectors using fastText: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec as glove/wiki.simple.vec
The data for style transfer is processed by appending the style values at the beginning of the sentence for source and target as <Style1> <Style2> .. <Style7> sentence. This can be found on https://drive.google.com/file/d/1vrBA9nhqDEaURVjzkp9xyN7avpCKZAbh/view .

__Baselines:__
runEmbDistRetrieve.sh and runModelAll.sh, runInferenceAll.sh runs the baselines for EmbDistRetrieve and Seq2Seq with attention.
Our Method: Copy Pointer and Decoder side self Attention
Create tmp/ directory in Style-Classification-Transfer/code/StyleTransfer/
./runModelAllPointer.sh contains the copy pointer method which was appended to the original Seq2Seq with attention.
Arguments for runModelPointer.sh
mode: ‘train’,’inference’ for training and inference times
preTrain: True if the model pretrains embeddings on formality style transfer data
NUM_PRETRAIN_EPOCHS: number of epochs for pretraining the embeddings
pointer: True if the model runs copy pointer method
NUM_EPOCHS: number of epochs to run the model on
problem: ‘STYLED’,’ethnic’ ‘gender’ ‘Country’ ‘edu’ ‘TOD’ ‘Politics’. We use STYLED in the case of transfer using all the styles.
decAttn: True if decoder side attention is enabled
Additional arguments in runInferenceAllPointer.sh
method: ‘beam’, ‘greedy’, ‘topk’ chooses the decoding method.
beamSIze chooses the size of beam in case of beam search and the value of k in case of topk sampling (default set to 3).