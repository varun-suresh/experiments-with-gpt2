"""
Train loop for sentence BERT
"""
import os
from tqdm import tqdm
from dataclasses import asdict
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from snliDataset import sentenceBERTDataset
from sentenceBERT import sentenceBERT
from bert_utils import dynamic_padding
from bert_config import BERTConfig, BERTTrainConfig



class Trainer:
    def __init__(self,train_set: sentenceBERTDataset, val_set: sentenceBERTDataset,model_config: BERTConfig, train_config:BERTTrainConfig):
        self.train_set = train_set
        self.val_set = val_set
        self.train_config = train_config
        self.model_config = model_config
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.writer = SummaryWriter(log_dir=self.train_config.out_dir)
    
    def load_model(self):
        if self.train_config.init_from == "resume":
            ckpt_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
            print(f"Resuming training from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=self.train_config.device)
            model_config = BERTConfig(**self.ckpt["model_config"])
            # Update some params
            model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
            model_config.checkpoint_path = self.model_config.checkpoint_path
            self.model_config = model_config
            self.model = sentenceBERT(self.model_config)
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model = sentenceBERT(self.model_config)
        self.model.to(self.train_config.device)


    def freeze_layers(self, N):
        for pn, p in self.model.named_parameters():
            if "embedding" in pn:
                p.requires_grad = False
            elif pn.split(".")[0] == "encoder" and int(pn.split(".")[2]) <= N:
                p.requires_grad = False
    
    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1,self.train_config.beta2))
        if self.train_config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])

    def train(self):
        self.load_model()
        self.load_optimizer()
        self.model.train()
        if self.train_config.freeze_layers > 0:
            self.freeze_layers(self.train_config.freeze_layers)

        if self.train_config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
            best_val_loss = self.ckpt["best_val_loss"]
        else:
            start_iter = 0
            best_val_loss = 1e9
        
        dl = DataLoader(self.train_set, batch_size=self.train_config.micro_batch_size,collate_fn=dynamic_padding,shuffle=True)
        accumulation_steps = self.train_config.batch_size // self.train_config.micro_batch_size

        for self.iter_num in tqdm(range(start_iter, self.train_config.max_iters)):
            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_config.grad_clip)
            
            if self.iter_num % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_losses()
                print(f"Step: {self.iter_num}\nTrain Loss: {losses['train']}\nValidation Loss:{losses['val']}")
                self.writer.add_scalar("Loss/train",losses["train"], self.iter_num)
                self.writer.add_scalar("Loss/val", losses["val"],self.iter_num)
                 
                if losses["val"] < best_val_loss or self.train_config.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if self.iter_num > start_iter:
                        ckpt = {"model": self.model.state_dict(),
                        "train_config": asdict(self.train_config),
                        "model_config": asdict(self.model_config),
                        "optimizer": self.optimizer.state_dict(),
                        "iter_num": self.iter_num,
                        "best_val_loss": best_val_loss}
                        output_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
                        print(f"Saving checkpoint to {output_path}")
                        if not os.path.exists(self.train_config.out_dir):
                            os.makedirs(self.train_config.out_dir)
                        torch.save(ckpt, output_path)
            
            batch = next(iter(dl))
            logits = self.model(batch["sentence_1"], batch["sentence_2"])
            loss = self.criterion(logits, batch["label"].to(self.train_config.device)) / (self.train_config.micro_batch_size * accumulation_steps)
            loss.backward()
 

    def estimate_losses(self):
        self.model.eval()
        train_dl = DataLoader(self.train_set, batch_size=self.train_config.micro_batch_size,collate_fn=dynamic_padding,shuffle=True)
        val_dl = DataLoader(self.val_set, batch_size=self.train_config.micro_batch_size,collate_fn=dynamic_padding,shuffle=True)
        train_loss = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(self.train_config.eval_iters):
                train_batch = next(iter(train_dl))
                train_logits = self.model(train_batch["sentence_1"], train_batch["sentence_2"])
                train_loss += self.criterion(train_logits, train_batch["label"].to(self.train_config.device))
                val_batch = next(iter(val_dl))
                val_logits =  self.model(val_batch["sentence_1"], val_batch["sentence_2"])
                val_loss += self.criterion(val_logits, val_batch["label"].to(self.train_config.device))
        
        losses = {}
        losses["train"] = train_loss / (self.train_config.eval_iters * self.train_config.micro_batch_size)
        losses["val"] = val_loss / (self.train_config.eval_iters * self.train_config.micro_batch_size)
        self.model.train()
        return losses
