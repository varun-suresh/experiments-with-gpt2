"""
Script to finetune GPT-2
"""
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import loralib as lora
from language_models.gpt import GPT
from language_models.utils.gpt_utils import dynamic_padding
from lib.baseTrainer import BaseTrainer
from lib.baseScheduler import CosineSchedulerWithWarmup
# torch.manual_seed(1367)

class Trainer(BaseTrainer):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        super(Trainer,self).__init__(config,train_set,val_set,test_set,criterion)
    
    def load_model(self):
        super().load_model()
        if self.config.init_from != "resume":
            self.model = GPT.from_pretrained(config=self.model_config).to(self.config.device)
        if self.config.freeze_layers > 0:
           self.freeze_layers(self.config.freeze_layers)

        if self.model_config.use_lora:
            lora.mark_only_lora_as_trainable(self.model)
        # Need to learn the classification layer. Explicitly set the gradient to True
        if self.model_config.binary_classification_head:
            self.model.classification_head.weight.requires_grad = True
     
    def freeze_layers(self,N):
        """
        Makes requires grad to false for the first N transformer blocks
        """
        for pn,p in self.model.named_parameters():
            if "wpe" in pn or "wte" in pn:
                p.requires_grad = False
            elif pn.split(".")[1] == "h" and int(pn.split(".")[2]) < N:
                p.requires_grad = False

    def load_optimizer_scheduler(self):
        param_dict = {pn:p for pn,p in self.model.named_parameters()}
        # Filter out all params that do not require grad
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # Create optim groups. Weight tensors in embeddings and attention blocks decay, biases and layernorms don't
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        self.optimizer = torch.optim.AdamW(optim_groups,lr=self.config.learning_rate,betas=(self.config.beta1,self.config.beta2))
        self.scheduler = CosineSchedulerWithWarmup(self.optimizer,lr=self.config.learning_rate,min_lr=self.config.min_lr,decay_iters=self.config.lr_decay_iters,warmup_iters=self.config.warmup_iters)
        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()
    
    def create_dataloader(self, dataset):
        return DataLoader(dataset, 
                        batch_size=self.config.micro_batch_size,
                        collate_fn=dynamic_padding,
                        shuffle=True)

    def run_inference(self,batch):
        return self.model(batch["input_ids"].to(self.config.device), batch["review_lens"].to(self.config.device))

    def calculate_subset_error(self, dataset):
        correct = 0
        dl = self.create_dataloader(dataset)
        threshold = 0.5
        for batch in tqdm(dl):
            with torch.no_grad():
                logits = self.run_inference(batch)
                predictions = F.sigmoid(logits)
                for label,pred in zip(batch['label'],predictions):
                    if label == 1 and pred >= threshold:
                        correct += 1
                    elif label == 0 and pred < threshold:
                        correct += 1
        error = 1 - correct/len(dataset)
        return error