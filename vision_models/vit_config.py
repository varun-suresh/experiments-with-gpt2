from dataclasses import dataclass


@dataclass
class VitConfig:
    d_emb: int = 192
    img_size: int = 32
    n_heads: int = 8
    n_blocks: int = 6
    n_classes: int = 10
    patch_size: int = 4
    num_patches: int = (img_size // patch_size) ** 2  # Assumes a square image
    dropout: float = 0.1
    cls_token: bool = True


@dataclass
class VitTrainConfig:
    # I/O
    out_dir: str = "out/vit"
    checkpoint_name: str = "vit_ckpt_cifar.pt"
    model_type: str = "vit"

    init_from: str = (
        "scratch"  # 'scratch' or 'resume' - it will resume from the latest checkpoint
    )
    always_save_checkpoint: bool = False

    # data
    batch_size: int = 512

    # AdamW optimizer
    learning_rate: float = 1e-4
    # max_iters:int = 50000
    epochs = 200
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # Cosine Scheduler
    min_lr: float = 1e-5
    decay_iters: int = 200
    warmup_iters: int = 1

    # device
    device: str = "cuda"

    # Gradient Accumulation
    micro_batch_size: int = 512

    freeze_layers: int = 0


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
    checkpoint_path: str = "vit/transforms/vit_ckpt_cifar.pt"
