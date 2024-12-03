"""
Script to finetune GPT-2
"""
import os
import math
from typing import Dict
from tqdm import tqdm
from dataclasses import asdict
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
from reviewsDataset import reviewsDataset
from gpt import GPT
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig 
from train_config import TrainConfig

# torch.manual_seed(1367)

class Trainer:
    def __init__(self,train_set: reviewsDataset,val_set: reviewsDataset,train_config:TrainConfig,model_config:GPTConfig):
        self.train_set = train_set
        self.val_set = val_set
        self.train_config = train_config
        self.model_config = model_config
        self.writer = SummaryWriter(log_dir=self.train_config.out_dir)
        self.iter_num = 0
    
    def get_lr(self):
        """
        Cosine learning rate with warmup
        """
        if self.iter_num < self.train_config.warmup_iters:
            return self.train_config.learning_rate * self.iter_num / self.train_config.warmup_iters
        if self.iter_num > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        decay_ratio = (self.iter_num - self.train_config.warmup_iters) / (self.train_config.lr_decay_iters - self.train_config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.train_config.min_lr + coeff * (self.train_config.learning_rate - self.train_config.min_lr)

    def load_model(self):
        if self.train_config.init_from == "resume":
            ckpt_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
            print(f"Resuming training from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path,map_location=self.train_config.device)
            model_config = GPTConfig(**self.ckpt["model_config"])
            model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
            model_config.checkpoint_path = self.model_config.checkpoint_path
            self.model_config = model_config
            self.model = GPT(self.model_config)
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model = GPT.from_pretrained(config=self.model_config) 
        if self.train_config.freeze_layers > 0:
            self.freeze_layers(self.train_config.freeze_layers)

        if self.model_config.use_lora:
            lora.mark_only_lora_as_trainable(self.model)
        # Need to learn the classification layer. Explicitly set the gradient to True
        if self.model_config.binary_classification_head:
            self.model.classification_head.weight.requires_grad = True
        self.model.to(self.train_config.device)
        if self.train_config.compile:
            print("Compiling the model..")
            self.model = torch.compile(self.model)


    def freeze_layers(self,N):
        """
        Makes requires grad to false for the first N transformer blocks
        """
        for pn,p in self.model.named_parameters():
            if "wpe" in pn or "wte" in pn:
                p.requires_grad = False
            elif pn.split(".")[1] == "h" and int(pn.split(".")[2]) < N:
                p.requires_grad = False

    def load_optimizer(self):
        self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay, 
                                                self.get_lr(),
                                                (self.train_config.beta1,self.train_config.beta2),
                                                self.train_config.device)

        if self.train_config.init_from =="resume":
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
    
 
    def train(self):

        self.load_model()
        self.load_optimizer()

        if self.train_config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
            best_val_loss = self.ckpt["best_val_loss"]
        else:
            start_iter = 0
            best_val_loss = 1e9

        dl = DataLoader(self.train_set, 
                        batch_size=self.train_config.micro_batch_size,
                        collate_fn=dynamic_padding,
                        shuffle=True)
        accumulation_steps = self.train_config.batch_size // self.train_config.micro_batch_size
        for self.iter_num in tqdm(range(start_iter,self.train_config.max_iters)):
            batch = next(iter(dl))
            if self.model_config.binary_classification_head:
                target = batch["labels"]
            else:
                target = batch["label_idxs"]
            logits, loss, _ = self.model(batch["input_ids"].to(self.train_config.device),
                                    batch["review_lens"].to(self.train_config.device),
                                    target=target.to(self.train_config.device))
     
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr()

            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_config.grad_clip)

            loss = loss / accumulation_steps 
            loss.backward()
            if self.iter_num % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"Step: {self.iter_num}\n Train Loss: {losses['train']}\nValidation Loss: {losses['val']}")

                self.writer.add_scalar("Loss/train",losses["train"],self.iter_num)
                self.writer.add_scalar("Loss/val",losses["val"],self.iter_num)
                for name,param in self.model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            self.writer.add_scalar(f"Grad/{name}",param.grad.norm(),self.iter_num)
                            self.writer.add_histogram(name, param, self.iter_num)
                            self.writer.add_histogram(f"{name}/grad",param.grad,self.iter_num)

                if losses["val"] < best_val_loss or self.train_config.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        ckpt = {"model": self.model.state_dict(),
                                    "train_config": asdict(self.train_config),
                                    "model_config": asdict(self.model_config),
                                    "optimizer":self.optimizer.state_dict(),
                                    "iter_num": self.iter_num,
                                    "best_val_loss": best_val_loss,
                                }
                        output_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name) 
                        print(f"Saving checkpoint to {output_path}") 
                        if not os.path.exists(self.train_config.out_dir):
                            os.makedirs(self.train_config.out_dir)
                        torch.save(ckpt,output_path)


    @torch.no_grad()
    def estimate_loss(self) -> Dict[str,float]:
        self.model.eval()
        train_dl = DataLoader(self.train_set,
                            batch_size=self.train_config.micro_batch_size,
                            collate_fn=dynamic_padding,
                            shuffle=True)
        val_dl = DataLoader(self.val_set,
                            batch_size=self.train_config.micro_batch_size,
                            collate_fn=dynamic_padding,
                            shuffle=True)
        train_loss = torch.zeros(self.train_config.eval_iters)
        val_loss = torch.zeros(self.train_config.eval_iters)
        for i in range(self.train_config.eval_iters):
            train_batch = next(iter(train_dl))
            val_batch = next(iter(val_dl))
            if self.model_config.binary_classification_head:
                target_train = train_batch['labels']
                target_val = val_batch['labels']
            else:
                target_train = train_batch['label_idxs']
                target_val = val_batch['label_idxs']

            _, train_loss[i],_ = self.model(train_batch['input_ids'].to(self.train_config.device), 
                                    train_batch['review_lens'].to(self.train_config.device),
                                    target=target_train.to(self.train_config.device))
            _, val_loss[i],_ = self.model(val_batch['input_ids'].to(self.train_config.device),
                            val_batch['review_lens'].to(self.train_config.device),
                            target=target_val.to(self.train_config.device))
    
        losses = {}
        losses["train"] = train_loss.mean()
        losses["val"] = val_loss.mean()
        self.model.train()
        return losses
