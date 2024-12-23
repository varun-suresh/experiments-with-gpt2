"""
Dataset for fine-tuning GPT-2 with SQuAD data
"""
from datasets import load_dataset
from torch.utils.data import Dataset

class squadDataset(Dataset):
    def __init__(self,split):
        super(squadDataset,self).__init__()
        self.dataset = load_dataset("rajpurkar/squad")[split]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
       sample = {
           "context": self.dataset[index]["context"],
           "question": self.dataset[index]["question"],
           "answer": self.dataset[index]["answers"]["text"][0],
       }
       return sample