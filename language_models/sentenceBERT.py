"""
Sentence BERT
"""
import numpy as np
import torch
import re
from torch import nn
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from language_models.bert import BERT
from language_models.bert_config import BERTConfig
from language_models.utils.bert_utils import sentence

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

    def encode(self,text,sentences_to_combine,overlap_size,batch_size=16):
        """
        open the file, run BERT to extract the embeddings. Return the embeddings
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text = re.sub(r'\[.*?\]','',text)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        overlapping_sentences = []
        for i in range(0,len(sentences),sentences_to_combine-overlap_size):
            end = min(len(sentences),i+sentences_to_combine)
            curr_sentence = " ".join(sentences[j] for j in range(i,end))
            overlapping_sentences.append(curr_sentence)
     

        encoded = tokenizer(overlapping_sentences,return_tensors="pt",padding=True)           
        encoded.attention_mask = encoded.attention_mask.bool()
        output_embeddings = np.zeros((encoded.input_ids.size(0),self.config.embedding_size))
        with torch.no_grad():
            for i in range(0,encoded.input_ids.size(0)):
                start = batch_size * i
                end = min(batch_size*(i+1),encoded.input_ids.size(0))
                embeddings = self.bert(encoded.input_ids[start:end,:].to(device),
                        encoded.token_type_ids[start:end,:].to(device),
                        encoded.attention_mask[start:end,:].to(device)).cpu().numpy()
                output_embeddings[start:end] = embeddings
        return output_embeddings,overlapping_sentences


if __name__ == "__main__":
    config = BERTConfig()
    sb = sentenceBERT(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text1 = tokenizer("Test sentence 1",return_tensors="pt")
    text2 = tokenizer("Test sentence 2",return_tensors="pt")

    sentence1 = sentence(**text1)
    sentence2 = sentence(**text2)

    print(sb(sentence1, sentence2))

        