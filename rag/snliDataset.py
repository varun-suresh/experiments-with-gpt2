# Dataloader for Stanford Natural Language Inference dataset

import os
import json

from torch.utils.data import Dataset

MAPPING = {"contradiction": 0, "entailment": 1, "neutral": 2}

class sentenceBERTDataset(Dataset):
    """
    Dataset to train and test sentence BERT
    """
    def __init__(self, split:str,cache_dir: str = "/home/varun/Downloads/",snli=True,mnli=True):
        assert split in ["train","test","dev"]
        self.data_file = []
        if snli:
            self.data_file.extend(open(os.path.join(cache_dir,"snli_1.0",f"snli_1.0_{split}.jsonl")).readlines())
        if mnli:
            if split == "train":
                self.data_file.extend(open(os.path.join(cache_dir,"multinli_1.0",f"multinli_1.0_{split}.jsonl")).readlines())
            elif split=="dev":
                self.data_file.extend(open(os.path.join(cache_dir,"multinli_1.0",f"multinli_1.0_{split}_matched.jsonl")).readlines())
                self.data_file.extend(open(os.path.join(cache_dir,"multinli_1.0",f"multinli_1.0_{split}_mismatched.jsonl")).readlines())

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self,idx:int):
        item = json.loads(self.data_file[idx])
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        label = item['gold_label']
        if label == "-":
            label = item['annotator_labels'][0]
        label = MAPPING[label]
        return {"sentence1": sentence1, "sentence2": sentence2,"label": label}

if __name__ == "__main__":
    sd = sentenceBERTDataset("train")
    for item in sd:
        print(f"Sentence1: {item['sentence1']}, Sentence2: {item['sentence2']}")
        break
    