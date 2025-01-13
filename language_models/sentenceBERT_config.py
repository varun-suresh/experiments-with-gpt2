from dataclasses import dataclass
from language_models.bert_config import BERTConfig, BERTTrainConfig

@dataclass
class sentenceBERTConfig(BERTConfig):
    model_type:str = "sentence-bert"
    n_classes:int = 3

@dataclass
class sentenceBERTTrainConfig(BERTTrainConfig):
    model_type:str = "sentence-bert"
    checkpoint_name: str = "sbert_ckpt_train_warmup.pt"
    warmup_iters:int = 12000
    step_size:int = 120000