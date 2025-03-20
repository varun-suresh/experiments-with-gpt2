from vision_models.resnet_config import ResNetCIFAR10Config
from vision_models.resnet import ResNetCifar
from vision_models.vit import VisionTransformer, VitConfig
from language_models.bert_config import BERTConfig
from language_models.sentenceBERT_config import sentenceBERTConfig
from language_models.bert import BERT
from language_models.gpt_config import GPTConfig
from language_models.gpt import GPT
from language_models.sentenceBERT import sentenceBERT
from sentiment_classification.sc_bert import sentimentClassificationBERT


class AvailableModels:
    def __init__(self):
        self.names = ["resnet-cifar", 
                                "BERT", 
                                "sentence-bert",
                                "gpt2",
                                "sc-bert",
                                "vit"]
    
        self.models = {"resnet-cifar": (ResNetCifar,ResNetCIFAR10Config), 
                    "bert": (BERT,BERTConfig), 
                    "sentence-bert": (sentenceBERT,sentenceBERTConfig),
                    "gpt2":(GPT,GPTConfig),
                    "sc-bert":(sentimentClassificationBERT,BERTConfig),
                    "vit": (VisionTransformer, VitConfig),
                    } 
    
    def get(self,model_name: str):
        model_def, model_config = self.models.get(model_name)
        return model_def, model_config