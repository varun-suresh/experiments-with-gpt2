"""
Train loop for sentence BERT
"""
from torch.utils.data.dataloader import DataLoader
from torch import nn
from snliDataset import sentenceBERTDataset
from language_models.utils.bert_utils import dynamic_padding
from language_models.sentenceBERT_config import sentenceBERTTrainConfig
from lib.baseTrainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        super(Trainer, self).__init__(config,train_set,val_set,test_set,criterion)

    def freeze_layers(self, N):
        for pn, p in self.model.named_parameters():
            if "embedding" in pn:
                p.requires_grad = False
            elif pn.split(".")[0] == "encoder" and int(pn.split(".")[2]) <= N:
                p.requires_grad = False

    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.config.micro_batch_size,collate_fn=dynamic_padding,shuffle=True)
    
    def run_inference(self,batch):
        return self.model(batch["sentence1"], batch["sentence2"]) 
          
if __name__ == "__main__":
    train_set = sentenceBERTDataset("train")
    val_set = sentenceBERTDataset("dev")
    test_set = sentenceBERTDataset("test")
    config = sentenceBERTTrainConfig()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer = Trainer(config,train_set,val_set,test_set,criterion)
    trainer.train()
