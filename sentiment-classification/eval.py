"""
Evaluate fine-tuned GPT-2 on IMDb movie reviews
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from reviewsDataset import reviewsDataset
from language_models.utils.gpt_utils import dynamic_padding
from language_models.gpt_config import GPTConfig
from eval_config import EvalConfig

from language_models.gpt import GPT

class Eval:
    def __init__(self,test_set: reviewsDataset,eval_config: EvalConfig, model_config: GPTConfig):
        self.test_set = test_set
        self.eval_config = eval_config
        self.model_config = model_config
        self.load_model()
    def load_model(self):
        if self.model_config.load_from_checkpoint:
            ckpt = torch.load(self.model_config.checkpoint_path,map_location=self.eval_config.device)
            model_config = GPTConfig(**ckpt["model_config"])
            model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
            model_config.checkpoint_path = self.model_config.checkpoint_path
            self.model_config = model_config
            self.model = GPT(self.model_config)
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model = GPT.from_pretrained(config=self.model_config)
        self.model.to(self.eval_config.device)
        if self.eval_config.compile:
            self.model = torch.compile(self.model)
        self.model.eval()

    def evaluate(self):
        if self.eval_config.subset:
            subset_range = torch.arange(0,len(self.test_set),self.eval_config.interval)
            dl = DataLoader(torch.utils.data.Subset(self.test_set,subset_range),batch_size=self.eval_config.batch_size,collate_fn=dynamic_padding)
        else:
            dl = DataLoader(self.test_set,batch_size=self.eval_config.batch_size,collate_fn=dynamic_padding)

        results_file = open(self.eval_config.results_path,"w")
        results_file.write("filename,length,label,prediction\n")
        for batch in tqdm(dl):
            with torch.no_grad():
                logits = self.model(batch["input_ids"].to(self.eval_config.device),batch["review_lens"].to(self.eval_config.device)).squeeze()
                if self.model_config.binary_classification_head:
                    predictions = F.sigmoid(logits)
                    for i, fname in enumerate(batch["fpaths"]):
                        results_file.write(f"{fname},{batch['lengths'][i]},{batch['label'][i]},{predictions[i].item()}\n")
                else:
                    sentiment_idx = self.test_set.get_pos_neg_indices()
                    for i, fname in enumerate(batch["fpaths"]):
                        pos = logits[i,sentiment_idx["positive"]]
                        neg = logits[i,sentiment_idx["negative"]]
                        if pos > neg:
                            prediction = 1
                        else:
                            prediction = 0
    
                        results_file.write(f"{fname},{batch['lengths'][i]},{batch['label'][i]},{prediction}\n")               

