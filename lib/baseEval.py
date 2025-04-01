"""
Base Class for Evaluating
"""
from abc import ABC,abstractmethod
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from lib.model_lib import AvailableModels

available_models = AvailableModels()
class BaseEval(ABC):
    def __init__(self,test_set,test_config):
        super(BaseEval, self).__init__()
        self.test_set = test_set
        self.test_config = test_config
        self.load_model()
    
    def load_model(self):
        ckpt = torch.load(self.test_config.checkpoint_path,map_location=self.test_config.device)
        model_name = ckpt["config"]["model_type"]
        model_def, model_config = available_models.get(model_name)
        model_config = model_config(**ckpt["model_config"])
        self.model_config = model_config
        self.model = model_def(self.model_config)
        self.model.load_state_dict(ckpt["model"])
        
        self.model.to(self.test_config.device)
        if self.test_config.compile:
            self.model = torch.compile(self.model)
        self.model.eval()
    
    def evaluate(self):
        if self.test_config.subset:
            subset_range = torch.arange(0,len(self.test_set),self.test_config.interval)
            dl = DataLoader(torch.utils.data.Subset(self.test_set,subset_range),batch_size=self.test_config.batch_size)
        else:
            dl = DataLoader(self.test_set,batch_size=self.test_config.batch_size)
        correct = 0
        for batch in tqdm(dl):
            with torch.no_grad():
                logits = self.model(batch["img"].to(self.test_config.device))
                predictions = torch.argmax(logits,dim=1)
                labels = torch.tensor(batch["label"]).to(self.test_config.device)
                correct += torch.eq(predictions,labels).sum()
        print(f"Accuracy: {correct/len(self.test_set)}\nError: {1-correct/len(self.test_set)}\nCorrect/Total: {correct}/{len(self.test_set)}")