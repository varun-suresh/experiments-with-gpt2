import torch
from torch import nn
from language_models.bert_config import BERTConfig
import loralib as lora

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        assert self.config.embedding_size % self.config.n_heads == 0
        self.query = nn.Linear(config.embedding_size, config.embedding_size)
        self.key = nn.Linear(config.embedding_size, config.embedding_size)
        self.value = nn.Linear(config.embedding_size, config.embedding_size)

    def setup_lora(self, r):
        self.query = lora.Linear(self.config.embedding_size,self.config.embedding_size, r)
        self.value = lora.Linear(self.config.embedding_size,self.config.embedding_size, r)

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
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1,self.config.n_heads,T,1)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask,dropout_p=self.config.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

class AttentionModule(nn.Module):
    def __init__(self,config):
        super(AttentionModule,self).__init__()
        self.config = config
        self.self = MultiHeadedAttention(config)
        self.output = nn.ModuleDict({
            "dense": nn.Linear(config.embedding_size, config.embedding_size),
            "LayerNorm": nn.LayerNorm(config.embedding_size,eps=config.layer_norm_eps),
        }) 
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self,x,attention_mask):
        x = x + self.resid_dropout(self.output.dense(self.self(x,attention_mask)))
        x = self.output.LayerNorm(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.config = config
        self.attention = AttentionModule(config) 

        self.intermediate = nn.ModuleDict({
            "dense": nn.Linear(config.embedding_size,4*config.embedding_size),
        })
        self.output = nn.ModuleDict({
            "dense": nn.Linear(4*config.embedding_size,config.embedding_size),
            "LayerNorm": nn.LayerNorm(config.embedding_size,eps=config.layer_norm_eps),
        })

    def forward(self, input,attention_mask):
        att_out = self.attention(input,attention_mask)
        intermediate = nn.functional.gelu(self.intermediate.dense(att_out))
        x = self.output.LayerNorm(att_out + self.output.dense(intermediate))
        return x


class BERT(nn.Module):
    def __init__(self,config):
        super(BERT,self).__init__()
        self.config = config
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(config.vocab_size, config.embedding_size),
            "position_embeddings": nn.Embedding(config.pretrained_block_size, config.embedding_size),
            "token_type_embeddings": nn.Embedding(2,config.embedding_size),
            "LayerNorm": nn.LayerNorm(config.embedding_size,eps=config.layer_norm_eps),
        })
        self.drop = nn.Dropout(self.config.dropout)
        self.encoder = nn.ModuleDict({
           "layer": nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers)),
        })
        self.pooler = nn.ModuleDict({
            "dense": nn.Linear(config.embedding_size,config.embedding_size),
            "activation": nn.Tanh()
        })
        if self.config.load_from_checkpoint:
            self._crop_block_size()
            if self.config.use_lora:
                self.setup_lora()


    def forward(self,input_ids,token_type_ids,attention_mask):
        device = input_ids.device
        b,t = input_ids.size()
        assert (
            t <= self.config.block_size 
        ), f"Sequence length {t} cannot be larger than the block size {self.config.block_size}"
        pos = torch.arange(0,t,dtype=torch.long,device=device)
        tok_emb = self.embeddings.word_embeddings(input_ids)
        pos_emb = self.embeddings.position_embeddings(pos)
        seg_emb = self.embeddings.token_type_embeddings(token_type_ids)
        x = self.drop(tok_emb + pos_emb + seg_emb)
        x = self.embeddings.LayerNorm(x)
        for block in self.encoder.layer:
            x = block(x,attention_mask)
        x = torch.mean(x,dim=1)
        # x = self.pooler.activation(self.pooler.dense(x[:,0,:]))
        return x

    def freeze_layers(self,N):
        for pn,p in self.named_parameters():
            if "embeddings" in pn:
                p.requires_grad = False
            elif pn.split(".")[0] == "encoder" and int(pn.split(".")[2]) < N:
                p.requires_grad = False

    def _crop_block_size(self):
        block_size = self.config.block_size
        self.embeddings.position_embeddings.weight = nn.Parameter(self.embeddings.position_embeddings.weight[:block_size])
        for block in self.encoder.layer:
            if hasattr(block.attention.self,"bias"):
                block.attention.self.bias = block.attention.self.bias[:,:,:block_size,:block_size]

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
        
        model._crop_block_size()
        if config.use_lora:
            model.setup_lora()
        return model
    
    def setup_lora(self):
        for i, block in enumerate(self.encoder.layer):
            if i in self.config.lora_layers:
                print(f"Setting up LoRA for layer {i}")
                block.attention.self.setup_lora(self.config.r)
if __name__ == "__main__":
    bert = BERT.from_pretrained(config=BERTConfig())