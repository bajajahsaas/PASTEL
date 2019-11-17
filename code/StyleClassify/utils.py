import os
import sys
import gensim
import numpy as np
import torch
from torch import nn
import logging

logging.basicConfig(level=logging.DEBUG, filename="LOG_FILENAME")
from transformers import BertModel, BertTokenizer

STYLE_ORDER = ['gender', 'country', 'age', 'ethnic', 'education', 'politics', 'tod']

def import_w2v_embeddings(filename, vocab=False, project=False):
    # import pdb; pdb.set_trace()
    if vocab is False:
        print('Loading gensim embedding from ',filename)
        glove_embedding = gensim.models.KeyedVectors.load_word2vec_format(
                filename, binary=True)
        return glove_embedding
    else:
        filename_vocab = filename+'.%s.%d' % (project,len(vocab))
        if not os.path.isfile(filename_vocab):
            fout = open(filename_vocab, 'w')
            for line in open(filename, 'r'):
                splitLine = line.strip().split(' ')
                word = splitLine[0]
                if word not in vocab:
                    continue
                fout.write('%s\n' % (line.strip()))
            fout.close()

        model = {}
        try:
            for line in open(filename_vocab, 'r'):
                splitLine = line.strip().split(' ')
                word = splitLine[0]
                if word not in vocab:
                    continue
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
        except Exception as e:
            print (e)
            import pdb
            pdb.set_trace()
        return model


def import_bert_embeddings(allsentences, vocab=False, project=False):
    # import pdb; pdb.set_trace()

    if vocab is False:
        print('Error in loading bert embedding')
        return None
    else:
        print("Building BERT embeddings for total sentences:", len(allsentences))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model = nn.DataParallel(model)
        model = model.to(device)

        emb = {}
        for iter, sent in enumerate(allsentences):
            bert_tokens_sentence = berttokenizer.encode(sent, add_special_tokens=True)
            with torch.no_grad():
                bert_embeddings = \
                    model(torch.tensor([bert_tokens_sentence]).to(device))[0].squeeze(0)
                f_emb_avg = torch.mean(bert_embeddings, axis=0).cpu().numpy()
                emb[sent] = f_emb_avg

            if iter%1000 == 0:
                print("BERT ready for ", iter, " sentences")

        return emb
