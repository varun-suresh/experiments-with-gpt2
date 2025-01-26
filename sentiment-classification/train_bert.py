"""
Train loop for sentiment classification using BERT
"""
from torch.utils.dataloader import DataLoader
from lib.baseTrainer import BaseTrainer
from language_models.utils.bert_utils import dynamic_padding_scbert
from language_models.bert import BERT

class Trainer(BaseTrainer):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        super(Trainer,self).__init__(config,train_set,val_set,test_set,criterion)

    def load_model(self):
        super().load_model()
        if self.config.init_from != "resume":
            self.model.bert = BERT.from_pretrained(config=self.model_config).to(self.config.device)
        if self.config.freeze_layers > 0:
            self.model.bert.freeze_layers(self.config.freeze_layers)
    
    def create_dataloader(self,dataset):
        return DataLoader(dataset,
                          batch_size=self.config.micro_batch_size,
                          collate_fn=dynamic_padding_scbert,
                          shuffle=True)

    def run_inference(self,batch):
        review = batch["review"]
        return self.model(review)
   