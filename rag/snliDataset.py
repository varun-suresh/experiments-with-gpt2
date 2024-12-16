# Dataloader for Stanford Natural Language Inference dataset

import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from bert_utils import sentence

MAPPING = {"contradiction": 0, "entailment": 1, "neutral": 2}

class snliDataset(Dataset):

    def __init__(self,split:str,cache_dir="/home/varun/Downloads/snli_1.0"):
        assert split in ('train','test','dev')
        self.split = split
        self.cache_dir = cache_dir
        data_file = open(os.path.join(cache_dir,f"snli_1.0_{split}.jsonl")).readlines()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = []
        keys_to_load = ["sentence1","sentence2","gold_label"]
        for i,line in enumerate(data_file):
            d = json.loads(line)
            if d["gold_label"] == "-":
                continue
            data_item = {}
            data_item = {k:d[k] for k in keys_to_load}
            data_item["label"] = MAPPING[data_item["gold_label"]]
            data_item |= tokenizer(d["sentence1"],d["sentence2"],return_tensors="pt")
            self.data.append(data_item)         

    def __len__(self):
        return len(self.data) 

    def get_mapping(self):
        return self.mapping
    
    def __getitem__(self,idx: int):
        return self.data[idx]

class snliEmbeddings(Dataset):

    def __init__(self,split:str):
        data_file = f"{split}_data.pt"
        if not os.path.exists(data_file):
            raise Exception(f"{data_file} does not exist, run prepare_data function first")
        self.data = torch.load(data_file)["data"]

    def __len__(self):
        return len(self.data) 

    def get_mapping(self):
        return MAPPING
    
    def __getitem__(self,idx: int):
        return {"embedding" : self.data[idx]["embedding"], "label": self.data[idx]["label"]}


class sentenceBERTDataset(Dataset):
    """
    Dataset to train and test sentence BERT
    """
    def __init__(self, split:str,cache_dir: str = "/home/varun/Downloads/snli_1.0/"):
        assert split in ["train","test","dev"]
        self.data_file = open(os.path.join(cache_dir,f"snli_1.0_{split}.jsonl")).readlines()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self,idx:int):
        item = json.loads(self.data_file[idx])
        sentence_1 = sentence(**self.tokenizer(item['sentence1'], return_tensors="pt"))
        sentence_2 = sentence(**self.tokenizer(item['sentence2'], return_tensors="pt"))
        label = MAPPING[item['gold_label']] 
        return {"sentence_1": sentence_1, "sentence_2": sentence_2,"label": label}

if __name__ == "__main__":
    sd = sentenceBERTDataset("train")
    for item in sd:
        print(f"Sentence1: {item['sentence_1']}, Sentence2: {item['sentence_2']}")
        break
    