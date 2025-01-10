"""
Train resnet on CIFAR-10
"""
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from cifar10 import cifar10
from resnet_config import ResNetCIFAR10TrainConfig
from lib.baseTrainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        super(Trainer,self).__init__(config,train_set,val_set,test_set,criterion)
    
    def freeze_layers(self):
        print(f"Freezing layers as specified in the config")
    
    def create_dataloader(self, dataset):
        dataloder = DataLoader(dataset,batch_size=self.config.micro_batch_size,shuffle=True)
        return dataloder
    
    def run_inference(self,batch):
        return self.model(batch["img"].to(self.config.device))

    def load_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                           lr=self.config.learning_rate,
                                            momentum=self.config.momentum,
                                            weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[32000,48000])
        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()


if __name__ == "__main__":
    all_train_data = cifar10("train")
    train_size = int(0.9*len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_set, val_set = random_split(all_train_data,[train_size,val_size])
    test_set = cifar10("test")
    test_set = random_split(test_set,[len(test_set)])[0]
    config = ResNetCIFAR10TrainConfig()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer = Trainer(config,train_set, val_set, test_set,criterion)
    trainer.train()