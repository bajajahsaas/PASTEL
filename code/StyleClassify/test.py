import torch
import logging
import numpy as np
from torch import nn

logging.basicConfig(level=logging.DEBUG, filename="LOG_FILENAME")
from transformers import BertModel, BertTokenizer
import os

sent = "He has tons of stuff to throw away."


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = nn.DataParallel(model)
model = model.to(device)
bert_tokens_sentence = berttokenizer.encode(sent, add_special_tokens=True)
with torch.no_grad():
    bert_embeddings = \
        model(torch.tensor([bert_tokens_sentence]).to(device))[0].squeeze(0)
    f_emb_avg = torch.mean(bert_embeddings, axis=0).cpu().numpy()
    f_value = np.array(f_emb_avg)