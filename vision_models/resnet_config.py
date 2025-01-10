
from dataclasses import dataclass
@dataclass
class ResNetCIFAR10Config:

    load_from_checkpoint : bool = False
    checkpoint_path : str = ""
    model_type = "resnet-cifar"

    # Depth: 6n+1:
    n: int = 9

@dataclass
class convConfig:
    input_channels: int
    output_channels: int
    kernel_size: int
    stride:int
    padding: int

@dataclass
class BottleNeckBlockConfig:
    input_channels: int
    intermediate_channels: int
    output_channels: int
    kernel_size: int = 1
    padding:int=0
    n_blocks: int = 1
    stride:int = 2

@dataclass
class maxpoolConfig:
    kernel_size: int
    stride: int=1
    padding:int = 0

@dataclass
class ResNet50Config:
    load_from_checkpoint: bool = False
    model_type: str = "resnet-imagenet"
    conv1 = convConfig(input_channels=3,output_channels=64,kernel_size=7,stride=2,padding=3)
    maxpool1 = maxpoolConfig(kernel_size=3,stride=2,padding=1)
    block2  = BottleNeckBlockConfig(input_channels=64,intermediate_channels=64,output_channels=256,n_blocks=3,stride=1)
    block3  = BottleNeckBlockConfig(input_channels=256,intermediate_channels=128,output_channels=512,n_blocks=4)
    block4  = BottleNeckBlockConfig(input_channels=512,intermediate_channels=256,output_channels=1024,n_blocks=6)
    block5 = BottleNeckBlockConfig(input_channels=1024,intermediate_channels=512,output_channels=2048,n_blocks=3)
    n_classes: int = 1000




@dataclass
class ResNetImageNetTrainConfig:
    # I/O
    out_dir: str = "imagenet/resnet-50"
    checkpoint_name: str = "resnet_ckpt_imagenet.pt"
    eval_interval: int = 2000
    eval_iters:int =  2000

    init_from: str = "scratch"
    always_save_checkpoint:bool = True

    # data
    batch_size: int=64



@dataclass
class ResNetCIFAR10TrainConfig:
    # I/O
    out_dir:str = "out/resnet-56-refactored"
    checkpoint_name: str = "resnet_ckpt_cifar.pt"
    eval_interval:int = 2000
    eval_iters:int = 100
    eval_size = 2000
    eval_only:bool = False
    model_type = "resnet-cifar"
    
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
    checkpoint_path: str = "out/resnet-56/resnet_ckpt_cifar.pt"