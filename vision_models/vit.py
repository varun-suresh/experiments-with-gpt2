"""
Implementation of Vision Transformer from scratch
"""
import torch
from torch import nn
import math

from vision_models.vit_config import VitConfig

class PatchEmbedding(nn.Module):
    def __init__(self,config):
        super(PatchEmbedding,self).__init__()
        self.conv = nn.Conv2d(3,config.d_emb,kernel_size=config.patch_size,stride=config.patch_size)
    
    def forward(self,x):
        # x -> (B,3,img_size,img_size)
        x = self.conv(x)
        x = x.flatten(2).transpose(1,2)
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
        self.resid_dropout = nn.Dropout(config.dropout)
    
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
        attn = self.resid_dropout(self.proj(attn))
        return attn

class FeedForwardBlock(nn.Module):
    def __init__(self,config):
        super(FeedForwardBlock,self).__init__()
        self.ff1 = nn.Linear(config.d_emb, 4*config.d_emb)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(4*config.d_emb,config.d_emb)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.gelu(self.ff1(x))
        x = self.resid_dropout(self.ff2(x))
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
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1,1,config.d_emb))
        self.position_embedding = nn.Parameter(torch.randn(1,config.num_patches+1,config.d_emb))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.classification_layer = nn.Linear(config.d_emb,config.n_classes) # CIFAR 10

    def forward(self,x):
        B,C,H,W = x.size()
        device = x.device
        patch_emb = self.patch_embedding(x) # (B,num_patches,d_emb)
        cls_tok = self.cls_token.expand(B,-1,-1) #(B,1,d_emb)
        patch_emb = torch.cat((cls_tok,patch_emb),dim=1) #(B,num_patches+1,d_emb)
        
        x = self.dropout(patch_emb + self.position_embedding)
        for block in self.blocks:
            x = block(x)
        return self.classification_layer(x[:,0])

if __name__ == "__main__":
    config = VitConfig()
    pe = PatchEmbedding(config)
    x = torch.rand(1,3,32,32)
    print(pe(x).size())