import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_components.embedding import DistributionEmbedding
from .transformer_components.positional_encoding import PositionalDistributionEncoding
from .transformer_components.transformer_block import DistTransformerBlock
from .transformer_components.lm_head import DistributionLMHead

class DistTransformer(nn.Module):
    """
    분포를 처리하는 트랜스포머 모델입니다.

    Args:
        vocab_size (int): 어휘 사전의 크기.
        d_model (int): 모델 차원.
        num_heads (int): 어텐션 헤드 수.
        d_ff (int): 피드포워드 신경망의 내부 차원.
        max_len (int): 최대 시퀀스 길이.
        num_blocks (int): 트랜스포머 블록의 수.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_len, num_blocks):
        super().__init__()
        self.embed = DistributionEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalDistributionEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([DistTransformerBlock(d_model, num_heads, d_ff) for _ in range(num_blocks)])
        self.lm_head = DistributionLMHead(d_model, vocab_size)

    def forward(self, tokens, optimizer):
        """
        입력 토큰에 대해 트랜스포머 모델을 실행하고 손실을 계산하여 역전파를 수행합니다.

        Args:
            tokens (torch.Tensor): 입력 토큰 시퀀스.
            optimizer: 모델 파라미터를 업데이트하는 옵티마이저.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 모델의 출력 로짓과 계산된 손실.
        """
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