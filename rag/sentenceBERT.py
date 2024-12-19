"""
Sentence BERT
"""
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer
import sys
sys.path.append("/home/varun/projects/experiments-with-gpt2/")
from bert import BERT
from bert_config import BERTConfig
from bert_utils import sentence
from time import time

# TODO : Add this in the config
device = "cuda"

class sentenceBERT(nn.Module):
    def __init__(self, config):
        super(sentenceBERT, self).__init__()
        self.config = config
        self.bert = BERT.from_pretrained(config)
        self.classification_layer = nn.Linear(3*config.embedding_size, config.n_classes)

    def forward(self,sentence_1,sentence_2):
        u = self.bert(sentence_1.input_ids.to(device),
                    sentence_1.token_type_ids.to(device),
                    sentence_1.attention_mask.to(device))
        v = self.bert(sentence_2.input_ids.to(device),
                    sentence_2.token_type_ids.to(device),
                    sentence_2.attention_mask.to(device))
        combined = torch.cat((u,v,torch.abs(u-v)),dim=1)
        output = self.classification_layer(combined)
        return output

    def encode(self,text,sentence_size,overlap_size,batch_size=16):
        """
        open the file, run BERT to extract the embeddings.
        Save the results in a vector database
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encoded = tokenizer(text,return_tensors="pt")
        enc_size = len(encoded.input_ids[0])
        idx = 0
        div_text = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        while idx < enc_size:
            end = min(enc_size,idx+sentence_size)
            curr_sentence = encoded.input_ids[0][idx:end]
            div_text.append(tokenizer.decode(curr_sentence))

            if len(curr_sentence) == sentence_size:
                input_ids.append(curr_sentence)
                attention_mask.append(encoded.attention_mask[0][idx:end])
                token_type_ids.append(encoded.token_type_ids[0][idx:end])
            else:
                zeros = torch.zeros(sentence_size-len(curr_sentence))
                input_ids.append(torch.cat((curr_sentence,zeros)))
                attention_mask.append(torch.cat((encoded.attention_mask[0][idx:end], zeros)))
                token_type_ids.append(torch.cat((encoded.token_type_ids[0][idx:end], zeros)))
    
            idx += sentence_size - overlap_size
            
        input_ids = torch.vstack(input_ids).int()
        attention_mask = torch.vstack(attention_mask).bool()
        token_type_ids = torch.vstack(token_type_ids).int()
        output_embeddings = np.zeros((input_ids.size(0),self.config.embedding_size))
        with torch.no_grad():
            for i in range(0,input_ids.size(0)):
                start = batch_size * i
                end = min(batch_size*(i+1),input_ids.size(0))
                embeddings = self.bert(input_ids[start:end,:].to(device),
                        token_type_ids[start:end,:].to(device),
                        attention_mask[start:end,:].to(device)).cpu().numpy()
                output_embeddings[start:end] = embeddings
        return output_embeddings, div_text 


if __name__ == "__main__":
    config = BERTConfig()
    sb = sentenceBERT(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text1 = tokenizer("Test sentence 1",return_tensors="pt")
    text2 = tokenizer("Test sentence 2",return_tensors="pt")

    sentence1 = sentence(**text1)
    sentence2 = sentence(**text2)

    print(sb(sentence1, sentence2))

        