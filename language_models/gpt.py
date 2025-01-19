import torch
from torch import nn
from torch.nn import functional as F
import math
import loralib as lora
import tiktoken
from language_models.gpt_config import GPTConfig

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        assert self.config.embedding_size % self.config.n_heads == 0
        self.c_attn = nn.Linear(self.config.embedding_size, 3*self.config.embedding_size)
        self.c_proj = nn.Linear(self.config.embedding_size, self.config.embedding_size)
        # Add dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)


    def setup_lora(self, r):
        self.c_attn = lora.MergedLinear(self.config.embedding_size, 3*self.config.embedding_size,r=r,enable_lora=[True,False,True])

    def forward(self, x):
        B, T, C = x.shape  # Batch Size, Block Size/ Sequence Length, Embedding Size
        q, k, v = self.c_attn(x).split(self.config.embedding_size, dim=2)
        q = q.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        k = k.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        if self.config.debug:
            att = (q @ k.transpose(-2,-1)) * (1.0 * math.sqrt(k.size(-1)))
            bias = torch.tril(torch.ones(self.config.block_size,self.config.block_size)).view(1,1,self.config.block_size,self.config.block_size)
            att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att,dim=-1)
            
            
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.config.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        # if self.config.debug:
            # return y, att
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.c_fc = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.embedding_size)
        self.attn = MultiHeadedAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_size)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class GPT(nn.Module):
    """
    Define the GPT-2 architecture and the ability to load the pretrained model
    """

    def __init__(self, config:GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.embedding_size),
                "wpe": nn.Embedding(self.config.pretrained_block_size, self.config.embedding_size),
                "drop": nn.Dropout(self.config.dropout),
                "h": nn.ModuleList(
                    TransformerBlock(self.config) for _ in range(self.config.n_layers)
                ),
                "ln_f": nn.LayerNorm(config.embedding_size),
            }
        )

        self.lm_head = nn.Linear(
            self.config.embedding_size, self.config.vocab_size, bias=False
        )
        self.transformer.wte.weight = (self.lm_head.weight)
        if config.binary_classification_head:
            self.setup_binary_classification_head()
        # Init weights:
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

        if self.config.load_from_checkpoint:
            self._crop_block_size()
            if self.config.use_lora:
                self.setup_lora()


        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self,non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _crop_block_size(self):
        block_size = self.config.block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def forward(self, idx,review_lens,target=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} is larger than the block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        att_out = []
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # To finetune, want to calculate the loss only on the last token
        if self.config.binary_classification_head:
            logits = self.classification_head(torch.stack([x[i,review_lens[i]-1,:] for i in range(len(review_lens))],dim=0))
            # if target is not None:
                # loss = F.binary_cross_entropy_with_logits(logits.squeeze(),target=target)
            # else:
                # loss = None
        else:
            logits = self.lm_head(torch.stack([x[i,[review_lens[i]-1],:] for i in range(len(review_lens))],dim=0))
            # if target is not None:
                # loss = F.cross_entropy(logits,target) 
            # else:
                # loss = None
        # return logits, loss, att_out
        return logits

    # def configure_optimizers(self,weight_decay,learning_rate,betas,device_type):
    #     param_dict = {pn:p for pn,p in self.named_parameters()}
    #     # Filter out all params that do not require grad
    #     param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
    #     # Create optim groups. Weight tensors in embeddings and attention blocks decay, biases and layernorms don't
    #     decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0},
    #         ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas=betas)
    #     return optimizer

    @classmethod
    def from_pretrained(cls, config:GPTConfig=GPTConfig()):
        """
        Downloads the Hugging Face model and copies the pre-trained weights on to the model defined here.
        """
        from transformers import GPT2LMHeadModel

        print(f"Loading pre-trained weights for {config.model_type}")
        # Load the pre-trained GPT-2 from Hugging Face 
        model = GPT(config) 
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Init a hugging face transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(config.model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        

        if config.binary_classification_head:
            assert len(sd_keys_hf) == len(sd_keys) - 2, f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys) - 2}"
        else:
            assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        model._crop_block_size()
        if config.use_lora:
            model.setup_lora()

        return model
    
    def setup_lora(self):
        for i, block in enumerate(self.transformer.h):
            if i in self.config.lora_layers:
                block.attn.setup_lora(self.config.r)

    def setup_binary_classification_head(self):
        self.classification_head = nn.Linear(self.config.embedding_size,1)

    @torch.no_grad()
    def generate(self, text, max_new_tokens:int, temp:float=0.8, top_k:int=None,device="cuda"):
        """
        Take a conditioning sequence and generate max_new_tokens number of tokes. Predictions are fed back in to the model each time
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        idx = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        idx = torch.tensor([idx],dtype=torch.long).to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _,_ = self(idx_cond,review_lens=torch.tensor([idx_cond.size(1)]).to(device))
            logits = logits[:,-1,:]/temp
             # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        
        return tokenizer.decode(idx[0].tolist())
    
if __name__ == "__main__":
    config = GPTConfig()
    model = GPT.from_pretrained()
