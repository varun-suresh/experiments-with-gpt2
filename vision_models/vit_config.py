from dataclasses import dataclass

@dataclass
class VitConfig: 
    d_emb = 192
    img_size = 32
    n_heads = 12
    n_blocks = 9
    n_classes = 10
    patch_size = 4
    num_patches = 64 # (img_size // patch_size)**2
    d_seq = (img_size // patch_size)**2 # Assumes a square image
    dropout = 0.2

@dataclass
class VitTrainConfig:
    # I/O
    out_dir:str = "out/vit"
    checkpoint_name: str = "vit_ckpt_cifar.pt"
    eval_interval:int = 2000
    eval_iters:int = 100
    eval_size = 2000
    eval_only:bool = False
    model_type:str = "vit"
    
    init_from:str = "scratch" # 'scratch' or 'resume' - it will resume from the latest checkpoint
    always_save_checkpoint:bool = False

    # data
    batch_size:int = 512

    # AdamW optimizer
    learning_rate:float = 0.001
    max_iters:int = 50000
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip:float = 1.0

    #device
    device:str = "cuda"

    # Gradient Accumulation
    micro_batch_size:int = 256

    step_size:int = 15000
    warmup_iters:int = 1000
    freeze_layers:int = 0

@dataclass
class VitTestConfig:
   # Device params
    device: str = "cuda"
    compile: bool = False

    # Dataset
    batch_size: int = 256
    subset: bool = False
    interval: int = 100

    # Model checkpoint
    checkpoint_path: str = "out/vit/vit_ckpt_cifar.pt"