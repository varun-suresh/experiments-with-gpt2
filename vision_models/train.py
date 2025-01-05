"""
Train resnet on CIFAR-10
"""
import os
from dataclasses import asdict
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from resnet import ResNetCifar
from cifar10 import cifar10
from resnet_config import ResNetCIFAR10Config, ResNetCIFAR10TrainConfig

class Trainer:
    def __init__(self,train_set,val_set,test_set,train_config,model_config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.train_config = train_config
        self.model_config = model_config
        self.iter_num = 0
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.writer = SummaryWriter(log_dir=train_config.out_dir)

    def load_model(self):
        if self.train_config.init_from == "resume":
            ckpt_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
            print(f"Resuming training from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=self.train_config.device)
            model_config = ResNetCIFAR10Config(**self.ckpt["model_config"])
            #Update some params
            model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
            model_config.checkpoint_path = self.model_config.checkpoint_path
            self.model_config = model_config
            self.model = ResNetCifar(self.model_config.n)
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model = ResNetCifar(self.model_config.n)
        self.model.to(self.train_config.device)

    def load_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                           lr=self.train_config.learning_rate,
                                            momentum=self.train_config.momentum,
                                            weight_decay=self.train_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[32000,48000])
        if self.train_config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()

    def train(self):
        self.load_model()
        self.load_optimizer_scheduler()

        dl = DataLoader(self.train_set,batch_size=self.train_config.micro_batch_size,shuffle=True)
        accumulation_steps = self.train_config.batch_size // self.train_config.micro_batch_size

        if self.train_config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
            best_val_loss = self.ckpt.get("best_val_loss",1e9)
        else:
            start_iter = 0
            best_val_loss = 1e9
        
        for self.iter_num in tqdm(range(start_iter,self.train_config.max_iters)):
            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_config.grad_clip)



            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_losses()
                test_error = self.calculate_test_error()
                print(f"Step:{self.iter_num}\nTrain Loss:{losses['train']}\nValidation Loss:{losses['val']}\nTest Error:{test_error}") 
                self.writer.add_scalar("Loss/train",losses["train"],self.iter_num)
                self.writer.add_scalar("Loss/val",losses["val"],self.iter_num)
                self.writer.add_scalar("Test Error",test_error,self.iter_num)

                for name,param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_scalar(f"Grad/{name}",param.grad.norm(),self.iter_num)
                        self.writer.add_histogram(name,param,self.iter_num)
                        self.writer.add_histogram(name,param.grad,self.iter_num)

                if losses["val"] < best_val_loss or self.train_config.always_save_checkpoint:
                    best_val_loss = losses["val"] 
                    if self.iter_num > start_iter:
                        ckpt = {
                            "model": self.model.state_dict(),
                            "train_config": asdict(self.train_config),
                            "model_config": asdict(self.model_config),
                            "iter_num": self.iter_num,
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                        }
                        output_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
                        if not os.path.exists(self.train_config.out_dir):
                            os.makedirs(self.train_config.out_dir)
                        print(f"Saving checkpoint to {output_path}")
                        torch.save(ckpt,output_path)
                
            if self.iter_num > start_iter and self.iter_num % accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()               

            batch = next(iter(dl))
            logits = self.model(batch["img"].to(self.train_config.device))
            loss = self.criterion(logits,batch["label"].to(self.train_config.device))
            loss.backward()


    def estimate_losses(self):
        self.model.eval()
        train_dl = DataLoader(self.train_set,batch_size=self.train_config.micro_batch_size,shuffle=True)
        val_dl = DataLoader(self.val_set,batch_size=self.train_config.micro_batch_size,shuffle=True)
        train_loss = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(self.train_config.eval_iters):
                train_batch = next(iter(train_dl))
                train_logits = self.model(train_batch["img"].to(self.train_config.device))
                train_loss += self.criterion(train_logits,torch.tensor(train_batch["label"]).to(self.train_config.device))
                val_batch = next(iter(val_dl))
                val_logits = self.model(val_batch["img"].to(self.train_config.device))
                val_loss += self.criterion(val_logits,val_batch["label"].to(self.train_config.device))
        
        losses = {}
        losses["train"] = train_loss/(self.train_config.eval_iters * self.train_config.micro_batch_size)
        losses["val"] = val_loss/(self.train_config.eval_iters * self.train_config.micro_batch_size)
        self.model.train()
        return losses

    def calculate_test_error(self):
        self.model.eval()
        test_dl = DataLoader(self.test_set,batch_size=self.train_config.micro_batch_size)
        correct = 0
        for batch in test_dl:
            with torch.no_grad():
                logits = self.model(batch["img"].to(self.train_config.device))
                predictions = torch.argmax(logits,dim=1)
                labels = torch.tensor(batch["label"]).to(self.train_config.device)
                correct += torch.eq(predictions,labels).sum()
        test_error = 1-correct/len(self.test_set)
        self.model.train()
        return test_error

if __name__ == "__main__":
    all_train_data = cifar10("train")
    train_size = int(0.8*len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_set, val_set = random_split(all_train_data,[train_size,val_size])
    train_config = ResNetCIFAR10TrainConfig()
    model_config = ResNetCIFAR10Config()
    trainer = Trainer(train_set, val_set, train_config, model_config)
    trainer.train()