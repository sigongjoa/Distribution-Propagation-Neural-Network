# dpnn_lib/models/dirichlet_denoiser.py

import torch
import torch.nn as nn

class DirichletDenoiser(nn.Module):
    """
    Dirichlet Diffusion 모델에서 디노이징을 수행하는 신경망입니다.
    alpha 벡터를 입력받아 디노이즈된 alpha 벡터를 예측합니다.

    Args:
        alpha_dim (int): alpha 벡터의 차원.
        hidden_dim (int): 은닉층의 차원.
        timesteps (int, optional): 총 타임스텝 수 (기본값: 101).
        time_embed_dim (int, optional): 시간 임베딩 차원 (기본값: 1).
    """
    def __init__(self, alpha_dim, hidden_dim, timesteps=101, time_embed_dim=1):
        super().__init__()
        self.t_embedding = nn.Embedding(num_embeddings=timesteps, embedding_dim=time_embed_dim)
        # 입력 차원 = alpha_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(alpha_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alpha_dim),
            nn.Softplus()
        )

    def forward(self, alpha_vector, t):
        """
        주어진 alpha 벡터와 timestep 정보를 사용하여 디노이즈된 alpha 벡터를 예측합니다.

        Args:
            alpha_vector (torch.Tensor): 노이즈가 추가된 alpha 벡터 (배치 크기, alpha_dim, 1 또는 배치 크기, alpha_dim).
            t (torch.Tensor): 현재 timestep 정보 (배치 크기).

        Returns:
            torch.Tensor: 디노이즈된 alpha 벡터 (배치 크기, alpha_dim, 1).
        """
        # alpha_vector: (batch, alpha_dim, 1) 또는 (batch, alpha_dim)
        if alpha_vector.dim() == 3 and alpha_vector.size(-1) == 1:
            # (batch, alpha_dim, 1) → (batch, alpha_dim)
            alpha_flat = alpha_vector.squeeze(-1)
        else:
            alpha_flat = alpha_vector  # 이미 (batch, alpha_dim)

        # t_embed: (batch, time_embed_dim)
        t_embed = self.t_embedding(t.long())

        # Ensure t_embed has the same batch size as alpha_flat
        if t_embed.shape[0] != alpha_flat.shape[0]:
            t_embed = t_embed.expand(alpha_flat.shape[0], -1)

        # 두 텐서를 concat하기 전 모두 2D로 만듦 → (batch, alpha_dim + time_embed_dim)
        combined = torch.cat([alpha_flat, t_embed], dim=-1)

        # 네트워크 통과 후 → (batch, alpha_dim)
        out = self.net(combined)

        # 다시 (batch, alpha_dim, 1) 형태로 복원
        return out.unsqueeze(-1)
