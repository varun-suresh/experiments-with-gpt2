"""
Train resnet on CIFAR-10
"""
import os
from dataclasses import asdict
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from resnet import ResNetCifar

class Trainer:
    def __init__(self,train_set,test_set,train_config,model_config):
        self.train_set = train_set
        self.test_set = test_set
        self.train_config = train_config
        self.model_config = model_config
        self.iter_num = 0
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=train_config.out_dir)

    def load_model(self):
        if self.train_config.init_from == "resume":
            raise NotImplementedError("Yet to be implemented")
        else:
            self.model = ResNetCifar(self.model_config.n)
        self.model.to(self.train_config.device)

    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.train_config.learning_rate,
                                           betas=(self.train_config.beta1,self.train_config.beta2))
        if self.train_config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
        self.optimizer.zero_grad()

    def train(self):
        self.load_model()
        self.load_optimizer()

        dl = DataLoader(self.train_set,batch_size=self.train_config.batch_size,shuffle=True)
        if self.train_config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
        else:
            start_iter = 0
        
        for self.iter_num in tqdm(range(start_iter,self.train_config.max_iters)):
            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.train_config.grad_clip)

            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_losses()
                print(f"Step:{self.iter_num}\nTrain Loss:{losses['train']}\nTest Loss:{losses['test']}") 
                self.writer.add_scalar("Loss/train",losses["train"],self.iter_num)
                self.writer.add_scalar("Loss/test",losses["test"],self.iter_num)
            
                if self.iter_num > start_iter:
                    ckpt = {
                        "model": self.model.state_dict(),
                        "train_config": asdict(self.train_config),
                        "model_config": asdict(self.model_config),
                        "iter_num": self.iter_num,
                        "optimizer": self.optimizer.state_dict(),
                    }
                    output_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
                    if not os.path.exists(self.train_config.out_dir):
                        os.makedirs(self.train_config.out_dir)
                    torch.save(ckpt,output_path)
            
            batch = next(iter(dl))
            logits = self.model(batch["img"])
            loss = self.criterion(logits,batch["label"].to(self.train_config.device))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
