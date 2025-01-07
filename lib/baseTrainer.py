"""
Base Trainer Class with default methods to
1. Load model
2. Load scheduler and optimizer
3. Run a training job
4. Calculate train and test error
"""
from abc import ABC,abstractmethod
from dataclasses import asdict
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from resnet_config import ResNetCIFAR10Config
from resnet import ResNetCifar
MODELS = {"resnet-cifar": ResNetCifar}
MODEL_CONFIGS = {"resnet-cifar": ResNetCIFAR10Config}
import os
import torch

class BaseTrainer(ABC):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        self.config = config
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.criterion = criterion
        self.writer = SummaryWriter(log_dir=config.out_dir)

    def load_model(self):
        """
        Loads the model based on the config
        """
        if self.train_config.init_from == "resume":
            ckpt_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
            print(f"Resuming training from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=self.train_config.device)
            model_config = MODEL_CONFIGS.get(self.config.model_type)(**self.ckpt["model_config"])
            #Update some params
            model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
            model_config.checkpoint_path = self.model_config.checkpoint_path
            self.model_config = model_config
            self.model = MODELS.get(self.config.model_type)(self.model_config.n)
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model = MODELS.get(self.config.model_type)(self.model_config.n)
        self.model.to(self.train_config.device)

    def load_optimizer(self):
        """
        By default, uses a AdamW optimizer
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1,self.train_config.beta2))
        if self.train_config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])

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
        
        dl = self.create_dataloader(self.train_set) 
        accumulation_steps = self.train_config.batch_size // self.train_config.micro_batch_size

        for self.iter_num in tqdm(range(start_iter, self.train_config.max_iters)):
            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_config.grad_clip)
            
            if self.iter_num % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_losses()
                test_error = self.calculate_test_error()
                print(f"Step: {self.iter_num}\nTrain Loss: {losses['train']}\nValidation Loss:{losses['val']}")
                self.writer.add_scalar("Loss/train",losses["train"], self.iter_num)
                self.writer.add_scalar("Loss/val", losses["val"],self.iter_num)
                self.writer.add_scalar("Test Error",test_error,self.iter_num)
                 
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
            model_output = self.run_inference(batch)
            loss = self.criterion(model_output, batch["label"].to(self.train_config.device)) / (self.train_config.micro_batch_size * accumulation_steps)
            loss.backward()
 
    def estimate_losses(self):
        self.model.eval()
        train_dl = self.create_dataloader(self.train_set)
        val_dl = self.create_dataloader(self.val_dl)
        train_loss = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(self.train_config.eval_iters):
                train_batch = next(iter(train_dl))
                train_output = self.run_inference(train_batch) 
                train_loss += self.criterion(train_output,torch.tensor(train_batch["label"]).to(self.train_config.device))
                val_batch = next(iter(val_dl))
                val_output = self.run_inference(val_batch) 
                val_loss += self.criterion(val_output,val_batch["label"].to(self.train_config.device))
        
        losses = {}
        losses["train"] = train_loss/(self.train_config.eval_iters * self.train_config.micro_batch_size)
        losses["val"] = val_loss/(self.train_config.eval_iters * self.train_config.micro_batch_size)
        self.model.train()
        return losses

    def calculate_test_error(self):
        """
        The default method assumes it is a multi-class classification problem
        """
        self.model.eval()
        test_dl = self.create_dataloader(self.test_set) 
        correct = 0
        for batch in test_dl:
            with torch.no_grad():
                model_output = self.run_inference(batch) 
                predictions = torch.argmax(model_output,dim=1)
                labels = torch.tensor(batch["label"]).to(self.train_config.device)
                correct += torch.eq(predictions,labels).sum()
        test_error = 1-correct/len(self.test_set)
        self.model.train()
        return test_error  