from torch.optim.lr_scheduler import LRScheduler

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