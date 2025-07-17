# dpnn_lib/models/poisson_denoiser.py

import torch
import torch.nn as nn

class PoissonDenoiser(nn.Module):
    """
    Poisson Diffusion 모델에서 디노이징을 수행하는 신경망입니다.
    rate 파라미터를 입력받아 디노이즈된 rate를 예측합니다.

    Args:
        hidden_dim (int): 은닉층의 차원.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # rate만 입력으로 받도록 단순화
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()   # rate는 양수여야 합니다
        )

    def forward(self, rate, t=None):
        """
        주어진 rate와 timestep 정보를 사용하여 디노이즈된 rate를 예측합니다.

        Args:
            rate (torch.Tensor): 노이즈가 추가된 rate (배치 크기, 1).
            t (torch.Tensor, optional): 현재 timestep 정보 (사용하지 않음).

        Returns:
            torch.Tensor: 디노이즈된 rate (배치 크기, 1).
        """
        # [batch] 또는 [batch,1] → [batch,1]
        if rate.dim() == 1:
            rate = rate.unsqueeze(-1)
        
        x = rate
        return self.net(x)
