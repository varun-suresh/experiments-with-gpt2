"""
Train Config
"""
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # I/O
    out_dir:str = "out"
    checkpoint_name: str = "ckpt_train.pt"
    eval_interval:int = 2000
    eval_iters:int = 100
    eval_size:int = 2000
    eval_only:bool = False
    always_save_checkpoint:bool = True
    init_from:str = "gpt2" # 'gpt2' or 'resume' - it will resume from the latest checkpoint
    compile:bool = False
    # data
    batch_size:int = 12

    # AdamW optimizer
    learning_rate:float = 6e-4
    max_iters:int = 60000
    weight_decay:float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    grad_clip:float = 5.0

    # Learning Rate scheduler : StepLR
    warmup_iters:int = 2000
    lr_decay_iters:int = 60000
    min_lr: float = 6e-5

    #device
    device:str = "cuda"
    model_type:str = "gpt2"
    # Gradient Accumulation
    micro_batch_size:int = 2   

    # Freeze layers when fine-tuning
    freeze_layers:int = 10