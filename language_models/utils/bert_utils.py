import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

@dataclass
class sentence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def __post_init__(self):
        self.attention_mask = self.attention_mask.bool()

def pad_sentences(s):
    input_ids = [x.input_ids[0] for x in s]
    token_type_ids = [x.token_type_ids[0] for x in s]
    attention_mask = [x.attention_mask[0] for x in s]
    # print(input_ids)
    input_ids_padded = pad_sequence(input_ids, batch_first=True,padding_value=0)
    token_type_ids_padded = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask,batch_first=True,padding_value=0)

    return {"input_ids": input_ids_padded, "token_type_ids":token_type_ids_padded, "attention_mask": attention_mask_padded}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def dynamic_padding(data):
    s1 = [tokenizer(item["sentence1"],return_tensors="pt") for item in data]
    s2 = [tokenizer(item["sentence2"],return_tensors="pt") for item in data]
    s1_padded = sentence(**pad_sentences(s1))
    s2_padded = sentence(**pad_sentences(s2))
    label = torch.tensor([item["score"] for item in data])
    # label = torch.tensor([item["label"] for item in data])

    return {"sentence1": s1_padded, "sentence2": s2_padded,"label": label}

def dynamic_padding_scbert(data,max_length=256):
    s1 = [tokenizer(item["review"],return_tensors="pt") for item in data]
    s1_padded = sentence(**pad_sentences(s1))
    label = torch.tensor([item["label"] for item in data])
    return {"review":s1_padded,"label":label}