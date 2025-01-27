"""
Train loop for sentiment classification using BERT
"""
from torch.utils.data.dataloader import DataLoader
from lib.baseTrainer import BaseTrainer
from language_models.utils.bert_utils import dynamic_padding_scbert
from language_models.bert import BERT
from tqdm import tqdm
import torch
import torch.nn.functional as F

class Trainer(BaseTrainer):
    def __init__(self,config,train_set,val_set,test_set,criterion):
        super(Trainer,self).__init__(config,train_set,val_set,test_set,criterion)

    def load_model(self):
        super().load_model()
        if self.config.init_from != "resume":
            self.model.bert = BERT.from_pretrained(config=self.model_config).to(self.config.device)
        if self.config.freeze_layers > 0:
            self.freeze_layers(self.config.freeze_layers)

    def create_dataloader(self,dataset):
        return DataLoader(dataset,
                          batch_size=self.config.micro_batch_size,
                          collate_fn=dynamic_padding_scbert,
                          shuffle=True)
    
    def freeze_layers(self,N):
        self.model.bert.freeze_layers(N)

    def run_inference(self,batch):
        review = batch["review"]
        return self.model(review,device=self.config.device)

    def calculate_subset_error(self, dataset):
        correct = 0
        dl = self.create_dataloader(dataset)
        threshold = 0.5
        for batch in tqdm(dl):
            with torch.no_grad():
                logits = self.run_inference(batch)
                predictions = F.sigmoid(logits)
                for label,pred in zip(batch['label'],predictions):
                    if label == 1 and pred >= threshold:
                        correct += 1
                    elif label == 0 and pred < threshold:
                        correct += 1
        error = 1 - correct/len(dataset)
        return error
if __name__ == "__main__":
    import torch
    from reviewsDataset import reviewsDataset
    from language_models.bert_config import BERTTrainConfig
    
    train_set, val_set = torch.utils.data.random_split(reviewsDataset("train"),[0.9,0.1])
    test_set = reviewsDataset("test") 
    config = BERTTrainConfig(model_type="sc-bert",freeze_layers=10,checkpoint_name="sc_bert.pt")
    criterion = torch.nn.BCELoss(reduction="sum")
    trainer = Trainer(config,train_set,val_set,test_set,criterion)
    trainer.train()