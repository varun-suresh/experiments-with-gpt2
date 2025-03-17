"""
Implementation of Vision Transformer from scratch
"""
from dataclasses import dataclass
import torch
from torch import nn
import math

@dataclass
class VitConfig: 
    d_emb = 256
    img_size = 32
    n_heads = 8
    n_blocks = 6
    n_classes = 10
    patch_size = 8
    d_seq = (img_size // patch_size)**2 # Assumes a square image

class TokenEmbedding(nn.Module):
    def __init__(self,config):
        super(TokenEmbedding,self).__init__()
        self.conv = nn.Conv2d(3,config.d_emb,kernel_size=config.patch_size,stride=config.patch_size)
        self.bn = nn.BatchNorm2d(config.d_emb)
    
    def forward(self,x):
        x = self.bn(self.conv(x))
        return x

class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self,config):
        super(MultiHeadedAttentionBlock,self).__init__()
        self.config = config
        self.query = nn.Linear(config.d_emb,config.d_emb)
        self.key = nn.Linear(config.d_emb,config.d_emb)
        self.value = nn.Linear(config.d_emb,config.d_emb)
        # Output projection
        self.proj = nn.Linear(config.d_emb,config.d_emb)
    
    def forward(self,x):
        B,ns,d = x.size() #Batch size, sequence length, embedding dimension(same as config.d_emb)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(B,ns,self.config.n_heads,d // self.config.n_heads).transpose(1,2) # (B,ns,n_heads,d_k) -> (B,n_heads,ns,d_k)
        k = k.view(B,ns,self.config.n_heads,d // self.config.n_heads).transpose(1,2) # (B,ns,n_heads,d_k) -> (B,n_heads,ns,d_k)
        v = v.view(B,ns,self.config.n_heads,d // self.config.n_heads).transpose(1,2) # (B,ns,n_heads,d_k) -> (B,n_heads,ns,d_k)
        attn = nn.functional.softmax(q@k.transpose(-2,-1)/math.sqrt(d/self.config.n_heads),dim=-1)@v
        attn = attn.transpose(1,2).contiguous().view(B,ns,d)
        attn = self.proj(attn)
        return attn

class FeedForwardBlock(nn.Module):
    def __init__(self,config):
        super(FeedForwardBlock,self).__init__()
        self.ff1 = nn.Linear(config.d_emb, 4*config.d_emb)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(4*config.d_emb,config.d_emb)
    
    def forward(self,x):
        x = self.gelu(self.ff1(x))
        x = self.ff2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super(TransformerBlock,self).__init__()
        self.mha = MultiHeadedAttentionBlock(config)
        self.ln1 = nn.LayerNorm(config.d_emb)
        self.ff = FeedForwardBlock(config)
        self.ln2 = nn.LayerNorm(config.d_emb)
    
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,config):
        super(VisionTransformer,self).__init__()
        self.config = config
        self.te = TokenEmbedding(config)
        self.pe = nn.Embedding(config.d_seq+1,config.d_emb) # +1 is for CLS token
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.classification_layer = nn.Linear(config.d_emb,config.n_classes) # CIFAR 10

    def forward(self,x):
        B,C,H,W = x.size()
        device = x.device
        n_patches = (H // self.config.patch_size) * (W // self.config.patch_size)
        token_embedding = torch.zeros((B,n_patches+1,self.config.d_emb)).to(device) # Additional CLS token
        token_embedding[:,1:] = self.te(x).reshape((B,self.config.d_emb,n_patches)).transpose(1,2)
        position_embedding = self.pe(torch.arange(0,self.config.d_seq+1).to(device))
        x = token_embedding + position_embedding
        for block in self.blocks:
            x = block(x)
        return self.classification_layer(x[:,0,:])

