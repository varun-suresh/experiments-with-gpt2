from dataclasses import dataclass
from typing import Tuple

MODELS = {
    "gpt2": dict(n_layers=12,n_heads=12,embedding_size=768),
    "gpt2-medium": dict(n_layers=24,n_heads=16,embedding_size=1024),
    "gpt2-large": dict(n_layers=36,n_heads=20,embedding_size=1024),
    "gpt2-xl": dict(n_layers=48,n_heads=25,embedding_size=1600),
}

@dataclass
class GPTConfig:

    vocab_size: int = 50257
    pretrained_block_size: int = 1024
    block_size:int = 256
    model_type: str = "gpt2" 
    # Load from a checkpoint
    load_from_checkpoint : bool = False
    checkpoint_path : str = ""

    # Use a binary classification head
    binary_classification_head:bool = False

    # Training specific params:
    # LoRA params
    use_lora:bool = True
    r:int = 8
    lora_layers: Tuple = (10,11)

    # Regularizaztion
    dropout: float = 0.1

    # For debugging
    debug:bool = False

    def __post_init__(self):
        if self.model_type not in MODELS:
            raise ValueError(f"Invalid model type {self.model_type}")
        for k,v in MODELS[self.model_type].items():
            setattr(self,k,v)