"""
Base Trainer Class with default methods to
1. Load model
2. Load scheduler and optimizer
3. Run a training job
4. Calculate train and test error
"""
import os
from abc import ABC,abstractmethod
from dataclasses import asdict
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from lib.baseScheduler import LRSchedulerWithWarmup
from lib.model_lib import AvailableModels

available_models = AvailableModels()
class BaseTrainer(ABC):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        self.config = config
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.criterion = criterion
        self.writer = SummaryWriter(log_dir=config.out_dir)
        self.load_model()
        self.load_optimizer_scheduler()
 
    def load_model(self):
        """
        Loads the model based on the config
        """
        model_def, model_config = available_models.get(self.config.model_type) 
        if self.config.init_from == "resume":
            ckpt_path = os.path.join(self.config.out_dir,self.config.checkpoint_name)
            print(f"Loading model from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=self.config.device)
            self.model_config = model_config(**self.ckpt["model_config"])
            self.model_config.load_from_checkpoint = True
            #Update some params
            # model_config.load_from_checkpoint = model_config.load_from_checkpoint
            # model_config.checkpoint_path = self.model_config.checkpoint_path
            # self.model_config = model_config
            self.model = model_def(self.model_config) 
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model_config = model_config()
            self.model = model_def(self.model_config)
        self.model.to(self.config.device)

    def load_optimizer_scheduler(self):
        """
        By default, uses a AdamW optimizer
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1,self.config.beta2))
        self.scheduler = LRSchedulerWithWarmup(self.optimizer,
                                               lr=self.config.learning_rate,
                                               step_size=self.config.step_size,
                                               gamma=0.1,
                                               warmup_iters=self.config.warmup_iters)
        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()
    

    @abstractmethod 
    def freeze_layers(self):
        """
        This method needs to be implemented in the train class for a particular model
        """
        pass

    @abstractmethod
    def create_dataloader(self,dataset,batch_size,collate_fn=None,shuffle=True):
        """
        Should return a dataloader. The dataloader must have a field label 
        """
        pass

    @abstractmethod
    def run_inference(self):
        """
        self.model(whatever inputs are needed for this particular model)
        """
        pass

    def train(self):
        print(f"Training..")
        self.model.train()
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters : {n_parameters}")
        if self.config.freeze_layers > 0:
            self.freeze_layers(self.config.freeze_layers)

        if self.config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
            best_val_loss = self.ckpt["best_val_loss"]
        else:
            start_iter = 0
            best_val_loss = 1e9
        
        dl = self.create_dataloader(self.train_set) 
        accumulation_steps = self.config.batch_size // self.config.micro_batch_size

        for self.iter_num in tqdm(range(start_iter, self.config.max_iters)):
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.grad_clip)
                       
            if self.iter_num % self.config.eval_interval == 0:
                losses = self.estimate_losses()
                errors = self.calculate_error()
                print(f"Step: {self.iter_num}\nTrain Loss: {losses['train']}\nValidation Loss:{losses['val']}")
                print(f"Train Error: {errors['train']}\nValidation Error: {errors['val']}\nTest Error: {errors['test']}")
                [self.writer.add_scalar(f"Loss/{split}", loss,self.iter_num) for split,loss in losses.items()]
                [self.writer.add_scalar(f"{split} error",error,self.iter_num) for split,error in errors.items()]
                 
                if losses["val"] < best_val_loss or self.config.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if self.iter_num > start_iter:
                        ckpt = {"model": self.model.state_dict(),
                        "config": asdict(self.config),
                        "model_config": asdict(self.model_config),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "iter_num": self.iter_num,
                        "best_val_loss": best_val_loss}
                        output_path = os.path.join(self.config.out_dir,self.config.checkpoint_name)
                        print(f"Saving checkpoint to {output_path}")
                        if not os.path.exists(self.config.out_dir):
                            os.makedirs(self.config.out_dir)
                        torch.save(ckpt, output_path)
            
            batch = next(iter(dl))
            model_output = self.run_inference(batch)
            loss = self.criterion(model_output, batch["label"].to(self.config.device)) / (self.config.micro_batch_size * accumulation_steps)
            loss.backward()

            if self.iter_num % accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
 
 
    def estimate_losses(self)->Dict[str,float]:
        self.model.eval()
        train_dl = self.create_dataloader(self.train_set)
        val_dl = self.create_dataloader(self.val_set)
        train_loss = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(self.config.eval_iters):
                train_batch = next(iter(train_dl))
                train_output = self.run_inference(train_batch)
                # print(f"Type of train output: {train_output.dtype}, Label: {train_batch['label'].dtype}") 
                train_loss += self.criterion(train_output,train_batch["label"].to(self.config.device))
                val_batch = next(iter(val_dl))
                val_output = self.run_inference(val_batch) 
                val_loss += self.criterion(val_output,val_batch["label"].to(self.config.device))
        
        losses = {}
        losses["train"] = train_loss/(self.config.eval_iters * self.config.micro_batch_size)
        losses["val"] = val_loss/(self.config.eval_iters * self.config.micro_batch_size)
        self.model.train()
        return losses

    def create_subset(self,dataset):
        subset = torch.utils.data.Subset(dataset,indices=torch.randint(high=len(dataset)-1,size=(self.config.eval_size,1)))
        return subset
    
    def calculate_subset_error(self,dataset):
        correct = 0
        dl = self.create_dataloader(dataset)

        for batch in dl:
            with torch.no_grad():
                model_output = self.run_inference(batch)
                predictions = torch.argmax(model_output,dim=-1)
                labels = torch.tensor(batch["label"]).to(self.config.device)
                correct += torch.eq(predictions,labels).sum()
        error = 1-correct/len(dataset)
        return error

    def calculate_error(self)->Dict[str,float]:
        """
        The default method assumes it is a multi-class classification problem
        """
        self.model.eval()

        train_subset = self.create_subset(self.train_set)
        test_subset = self.create_subset(self.test_set)
        val_subset = self.create_subset(self.val_set)

        pairs = [("train",train_subset),("val",val_subset),("test",test_subset)]

        errors = {}
        for split,subset in pairs:
            errors[split] = self.calculate_subset_error(subset) 
        self.model.train()
        return errors 