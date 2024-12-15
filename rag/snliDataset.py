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
    def __init__(self, split:str, cache_dir:str = "/home/varun/Downloads/snli_1.0/",recreate:bool=False):
        assert split in ["train", "test", "dev"]
        data_file = open(os.path.join(cache_dir,f"snli_1.0_{split}.jsonl")).readlines()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cache_fpath = os.path.join(cache_dir,f"sentenceBERT_{split}.pt")
        if not os.path.exists(cache_fpath) or recreate:
            self.data = []
            for i,line in enumerate(data_file):
                d = json.loads(line)
                if d["gold_label"] == "-":
                    continue
                self.data.append({"label": MAPPING[d["gold_label"]],
                "sentence_1": sentence(**tokenizer(d["sentence1"],return_tensors="pt")),
                "sentence_2": sentence(**tokenizer(d["sentence2"],return_tensors="pt"))})
            torch.save({"data": self.data}, cache_fpath)
        else:
            print(f"Loading extracted data from {cache_fpath}")
            self.data = torch.load(cache_fpath)["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx:int):
        return self.data[idx]

if __name__ == "__main__":
    sd = snliDataset("dev")
    for item in sd:
        print(f"Sentence1: {item['sentence1']}, Sentence2: {item['sentence2']}")
        break
    