import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_components.embedding import DistributionEmbedding
from .transformer_components.positional_encoding import PositionalDistributionEncoding
from .transformer_components.transformer_block import DistTransformerBlock
from .transformer_components.lm_head import DistributionLMHead

class DistTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_len, num_blocks):
        super().__init__()
        self.embed = DistributionEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalDistributionEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([DistTransformerBlock(d_model, num_heads, d_ff) for _ in range(num_blocks)])
        self.lm_head = DistributionLMHead(d_model, vocab_size)

    def forward(self, tokens, optimizer):
        # 1) 임베딩
        dist = self.embed(tokens)                           # GaussianDistribution
        dist = self.pos_enc(dist)                           # add positional dist

        # 2) 블록 스택
        for block in self.blocks:
            dist = block(dist)

        # 3) LM head
        logits = self.lm_head(dist)                         # (B,L,V)
        loss   = F.cross_entropy(logits.view(-1,logits.shape[-1]), tokens.view(-1))

        # 4) 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return logits, loss