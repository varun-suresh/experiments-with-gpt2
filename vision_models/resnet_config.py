
from dataclasses import dataclass
@dataclass
class ResNetConfig:

    load_from_checkpoint : bool = False
    checkpoint_path : str = ""
    model_type = "resnet-cifar"

    # Depth: 6n+1:
    n: int = 3

@dataclass
class ResNetTrainConfig:
    # I/O
    out_dir:str = "out/resnet"
    checkpoint_name: str = "resnet_ckpt_cifar.pt"
    eval_interval:int = 2000
    eval_iters:int = 100
    eval_only:bool = False
    
    init_from:str = "scratch" # 'scratch' or 'resume' - it will resume from the latest checkpoint
    always_save_checkpoint:bool = True

    # data
    batch_size:int = 128

    # AdamW optimizer
    learning_rate:float = 0.1
    max_iters:int = 64000
    momentum: float = 0.9
    weight_decay: float = 0.0001
    grad_clip:float = 1.0

    # # Learning Rate scheduler : StepLR
    # warmup_iters:int = 2000
    # lr_decay_iters:int = 40000
    # min_lr: float = 6e-5

    #device
    device:str = "cuda"

    # Gradient Accumulation
    micro_batch_size:int = 128 

    # Freeze layers when fine-tuning
    freeze_layers:int = 0

@dataclass
class ResNetTestConfig:

    # Device params
    device: str = "cuda"
    compile: bool = False

    # Dataset
    batch_size: int = 64
    subset: bool = False
    interval: int = 100

    # Model checkpoint
    checkpoint_path: str = "out/resnet/resnet_ckpt_cifar.pt"