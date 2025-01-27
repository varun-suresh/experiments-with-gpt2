"""
BERT for sentiment classification : Model definition
"""

import torch
from torch import nn
from transformers import BertTokenizer
from language_models.bert import BERT

class sentimentClassificationBERT(nn.Module):
    def __init__(self,config):
        super(sentimentClassificationBERT,self).__init__()
        self.config = config

        self.bert = BERT(config)
        self.classification_head = nn.Linear(self.config.embedding_size,1)
    
    def forward(self,input):
        x = self.bert(input.input_ids,
                      input.token_type_ids,
                      input.attention_mask)
        x = self.classification_head(x)
        return x