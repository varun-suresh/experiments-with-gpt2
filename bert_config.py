from dataclasses import dataclass
from typing import Tuple



@dataclass
class BERTConfig:

    vocab_size: int = 30522
    embedding_size: int = 768
    n_heads:int = 12
    n_layers:int = 12
    block_size:int = 512
    # Load from a checkpoint
    load_from_checkpoint : bool = False
    checkpoint_path : str = ""
    model_type = "BERT"
    is_causal:bool = False
    layer_norm_eps: float = 1e-12
    # # Training specific params:
    # # LoRA params
    # use_lora:bool = True
    # r:int = 8
    # lora_layers: Tuple = (10,11)

    # Regularizaztion
    dropout: float = 0.1

    # For debugging
    # debug:bool = False

    # def __post_init__(self):
    #     if self.model_type not in MODELS:
    #         raise ValueError(f"Invalid model type {self.model_type}")
    #     for k,v in MODELS[self.model_type].items():
    #         setattr(self,k,v)