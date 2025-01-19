from torch.optim.lr_scheduler import LRScheduler
import math

class LRSchedulerWithWarmup(LRScheduler):
    def __init__(self, optimizer,lr:float,step_size:int,gamma:float,warmup_iters:int):
        self.step_size = step_size
        self.gamma = gamma
        self.lr = lr
        self.warmup_iters = warmup_iters
        super().__init__(optimizer=optimizer,last_epoch=-1,verbose="deprecated")
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [self.lr * (self.last_epoch+1)/self.warmup_iters for group in self.optimizer.param_groups]
        if self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

class CosineSchedulerWithWarmup(LRScheduler):
    def __init__(self,optimizer,lr:float,min_lr:float,decay_iters:int,warmup_iters:int):
        self.lr = lr
        self.min_lr = min_lr
        self.decay_iters = decay_iters
        self.warmup_iters = warmup_iters
        super().__init__(optimizer=optimizer,last_epoch=-1)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [self.lr * (self.last_epoch+1) / self.warmup_iters for group in self.optimizer.param_groups]
        if self.last_epoch > self.decay_iters:
            return [self.min_lr for group in self.optimizer.param_groups]
        decay_ratio = (self.last_epoch - self.warmup_iters)/(self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio))
        return [self.min_lr + coeff * (self.lr - self.min_lr) for group in self.optimizer.param_groups]


