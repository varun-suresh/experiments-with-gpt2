import torch
from torch import nn
from torch.nn import functional as F
import math
from gpt import TransformerBlock
from bert_config import BERTConfig


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        assert self.config.embedding_size % self.config.n_heads == 0
        self.query = nn.Linear(config.embedding_size, config.embedding_size)
        self.key = nn.Linear(config.embedding_size, config.embedding_size)
        self.value = nn.Linear(config.embedding_size, config.embedding_size)

    def forward(self, x,attention_mask):
        B, T, C = x.shape  # Batch Size, Block Size/ Sequence Length, Embedding Size
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        k = k.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2) 
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(3).repeat(1,self.config.n_heads,1,T)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask)
        # y = torch.nn.functional.scaled_dot_product_attention(q,k,v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y

class AttentionModule(nn.Module):
    def __init__(self,config):
        super(AttentionModule,self).__init__()
        self.config = config
        self.self = MultiHeadedAttention(config)
        self.output = nn.ModuleDict({
            "dense": nn.Linear(config.embedding_size, config.embedding_size),
            "LayerNorm": nn.LayerNorm(config.embedding_size),
        })
    
    def forward(self,x,attention_mask):
        x = x + self.output.dense(self.self(x,attention_mask))
        x = self.output.LayerNorm(x)
        return x

class FFNIntermediate(nn.Module):
    def __init__(self,config):
        super(FFNIntermediate,self).__init__()
        self.dense = nn.Linear(config.embedding_size, 4*config.embedding_size)
    
    def forward(self,x):
        return nn.functional.gelu(self.dense(x))

class FFNOutput(nn.Module):
    def __init__(self, config):
        super(FFNOutput,self).__init__()
        self.dense = nn.Linear(4*config.embedding_size,config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
    
    def forward(self,input,x):
        return self.LayerNorm(input + self.dense(x))
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.config = config
        self.attention = AttentionModule(config) 
        self.intermediate = FFNIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, x,attention_mask):
        att_out = self.attention(x,attention_mask)
        z = self.intermediate(att_out)
        x = self.output(att_out,z)   
        return x


class BERT(nn.Module):
    def __init__(self,config):
        super(BERT,self).__init__()
        self.config = config
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(config.vocab_size, config.embedding_size),
            "position_embeddings": nn.Embedding(config.block_size, config.embedding_size),
            "token_type_embeddings": nn.Embedding(2,config.embedding_size),
            "LayerNorm": nn.LayerNorm(config.embedding_size),
        })
        self.encoder = nn.ModuleDict({
           "layer": nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers)),
        })
        self.pooler = nn.ModuleDict({
            "dense": nn.Linear(config.embedding_size,config.embedding_size),
        })

    def forward(self,input_ids,token_type_ids,attention_mask,target=None):
        device = input_ids.device
        b,t = input_ids.size()
        assert (
            t <= self.config.block_size 
        ), f"Sequence length {t} cannot be larger than the block size {self.config.block_size}"
        pos = torch.arange(0,t,dtype=torch.long,device=device)
        tok_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(pos)
        seg_emb = self.embeddings.token_type_embeddings(token_type_ids)
        x = self.embeddings.LayerNorm(tok_emb + pos_emb + seg_emb)
        for block in self.encoder.layer:
            x = block(x,attention_mask)
        # Return the [CLS] hidden state of the last layer
        # return torch.mean(x,dim=1)
        return x

        
    @classmethod
    def from_pretrained(cls,config):
        """
        Download the pre-trained weights from Hugging Face and copy the weights to the model defined here.
        """

        from transformers import BertModel
        print(f"Loading pre-trained weights for {config.model_type}")
        model = BERT(config)
        sd = model.state_dict()
        
       # Initialize a HF model
        model_hf = BertModel.from_pretrained('bert-base-uncased')
        sd_hf = model_hf.state_dict()

        assert len(sd_hf.keys()) == len(sd.keys())
        for k, v in sd_hf.items():
            assert k in sd
            assert v.size() == sd[k].size()

        for k,v in sd_hf.items():
           with torch.no_grad():
                sd[k].copy_(v)
        return model

if __name__ == "__main__":
    bert = BERT.from_pretrained(config=BERTConfig())