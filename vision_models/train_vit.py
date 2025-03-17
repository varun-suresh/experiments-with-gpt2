from dataclasses import dataclass
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from cifar10 import cifar10
from lib.baseTrainer import BaseTrainer

@dataclass
class VitTrainConfig:
    # I/O
    out_dir:str = "out/vit"
    checkpoint_name: str = "vit_ckpt_cifar.pt"
    eval_interval:int = 2000
    eval_iters:int = 100
    eval_size = 2000
    eval_only:bool = False
    model_type = "vit"
    
    init_from:str = "scratch" # 'scratch' or 'resume' - it will resume from the latest checkpoint
    always_save_checkpoint:bool = True

    # data
    batch_size:int = 128

    # AdamW optimizer
    learning_rate:float = 6e-3
    max_iters:int = 30000
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip:float = 1.0

    #device
    device:str = "cuda"

    # Gradient Accumulation
    micro_batch_size:int = 64 

    step_size:int = 30000
    warmup_iters:int = 10
    freeze_layers:int = 0

class Trainer(BaseTrainer):
    def __init__(self, config,train_set,val_set,test_set,criterion):
        super(Trainer,self).__init__(config,train_set,val_set,test_set,criterion)

    def create_dataloader(self,dataset):
        dataloader = DataLoader(dataset,batch_size=self.config.micro_batch_size,shuffle=True)
        return dataloader
    
    def freeze_layers(self):
        print("Not implemented yet")

    def run_inference(self,batch):
        return self.model(batch["img"].to(self.config.device))
    


if __name__ == "__main__":
    all_train_data = cifar10("train")
    train_size = int(0.9*len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_set, val_set = random_split(all_train_data,[train_size,val_size])
    test_set = cifar10("test")
    test_set = random_split(test_set,[len(test_set)])[0]
    config = VitTrainConfig()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer = Trainer(config,train_set, val_set, test_set,criterion)
    trainer.train()