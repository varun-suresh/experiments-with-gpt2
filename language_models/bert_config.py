from dataclasses import dataclass
from typing import Tuple


@dataclass
class BERTConfig:
    vocab_size: int = 30522
    embedding_size: int = 768
    n_heads: int = 12
    n_layers: int = 12
    pretrained_block_size: int = 512
    block_size: int = 256
    # Load from a checkpoint
    load_from_checkpoint: bool = False
    checkpoint_path: str = ""
    model_type: str = "bert"
    is_causal: bool = False
    layer_norm_eps: float = 1e-12
    # Training specific params:
    # LoRA params
    use_lora: bool = True
    r: int = 8
    lora_layers: Tuple = (10, 11)

    # Regularizaztion
    dropout: float = 0.1

    # For debugging
    # debug:bool = False


@dataclass
class BERTTrainConfig:
    # I/O
    out_dir: str = "out"
    model_type: str = "sc-bert"
    checkpoint_name: str = "bert_ckpt_train_warmup.pt"
    eval_interval: int = 2000
    eval_size: int = 2000
    eval_iters: int = 100
    eval_only: bool = False

    init_from: str = (
        "bert"  # 'bert' or 'resume' - it will resume from the latest checkpoint
    )
    always_save_checkpoint: bool = True

    # data
    batch_size: int = 16

    # AdamW optimizer
    learning_rate: float = 2e-5
    max_iters: int = 5000
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # # Learning Rate scheduler : StepLR
    warmup_iters: int = 12000
    step_size: int = 120000
    # lr_decay_iters:int = 40000
    # min_lr: float = 6e-5

    # device
    device: str = "cuda"

    # Gradient Accumulation
    micro_batch_size: int = 8

    # Freeze layers when fine-tuning
    freeze_layers: int = 0
