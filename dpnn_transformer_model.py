import torch
import torch.nn as nn
import torch.nn.functional as F

from dpnn.dists.gaussian import GaussianDiag
from dpnn.layers.blocks import DistTransformerBlock
from dpnn.core.config import preset_config, Preset, DistConfig

# (임시) dpnn_lib positional encoding이 텐서를 받도록 수정됨
from dpnn_lib.distributions.transformer_components.positional_encoding import     PositionalDistributionEncoding

# 호환 shim (분포<->텐서 왕복) - PositionalDistributionEncoding이 텐서를 요구할 경우
def posenc_shim(pos_enc, dist: GaussianDiag) -> GaussianDiag:
    # pos_enc가 텐서를 입력으로 받고 텐서를 출력한다고 가정
    y = pos_enc(dist.mean())  # 텐서 인풋 가정
    # 출력 텐서를 다시 GaussianDiag 분포로 변환
    return GaussianDiag.from_tensor(y, init_std=dist.scale.mean().item()) # 이전 분포의 평균 스케일 사용

class DPNNTransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers,
                 dropout=0.1, max_len=2048, preset: Preset = Preset.FAST, k_top: int = 8):
        super().__init__()
        self.ntoken = ntoken
        self.d_model = ninp
        self.embed = nn.Embedding(ntoken, ninp) # Use standard embedding
        self.pos_enc = PositionalDistributionEncoding(max_len, ninp)
        # dpnn 블록 스택
        self.blocks = nn.ModuleList([
            DistTransformerBlock(ninp, nhead, nhid, resample_every=2,
                                 k_top=k_top) # block_index는 현재 사용되지 않으므로 제거
            for _ in range(nlayers)
        ])
        # 비교 실험용: 텐서 헤드
        self.lm_head = nn.Linear(ninp, ntoken)
        # 블록에 넘겨줄 설정을 모델이 보관
        self.cfg: DistConfig = preset_config(preset)

    def forward(self, src: torch.Tensor):
        """
        src: (B, L) 토큰
        return: (B, L, vocab)
        """
        x = self.embed(src)                      # (B, L, D) tensor
        dist = GaussianDiag.from_tensor(x, init_std=0.1)  # 텐서→분포
        
        # Positional Encoding: pos_enc는 텐서를 입력으로 받도록 수정되었으므로 dist.mean() 전달
        dist.loc = self.pos_enc(dist.loc) # pos_enc의 출력을 dist의 loc에 할당

        total = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # total_blocks 인자는 DistTransformerBlock에서 사용되지 않으므로 제거
            dist = block(dist, self.cfg)

        logits = self.lm_head(dist.mean())       # 분포→텐서
        return logits
