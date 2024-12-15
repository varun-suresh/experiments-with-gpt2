"""
Sentence BERT
"""
import torch
from torch import nn
from typing import List
from transformers import BertTokenizer
import sys
sys.path.append("/home/varun/projects/experiments-with-gpt2/")
from bert import BERT
from bert_config import BERTConfig
from bert_utils import sentence

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

if __name__ == "__main__":
    config = BERTConfig()
    sb = sentenceBERT(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text1 = tokenizer("Test sentence 1",return_tensors="pt")
    text2 = tokenizer("Test sentence 2",return_tensors="pt")

    sentence1 = sentence(**text1)
    sentence2 = sentence(**text2)

    print(sb(sentence1, sentence2))

        