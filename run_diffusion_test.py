
import sys
import os

# Add the parent directory of dpnn_lib to the Python path
sys.path.insert(0, os.path.abspath('/mnt/d/progress/DPNN'))

import torch
import torch.optim as optim
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.diffusion.forward import forward_step
from dpnn_lib.diffusion.denoise import DistDiffusionBlock
from dpnn_lib.diffusion.loss import loss_fn
from dpnn_lib.diffusion.sampler import DiffusionSampler

# 1. 하이퍼파라미터 설정
d_model = 256      # 모델 차원
num_heads = 4      # 어텐션 헤드 수
d_ff = 1024        # 피드포워드 신경망의 내부 차원
T = 100            # 총 타임스텝 수
var_weight = 1.0   # 분산 손실 가중치

# 2. 모델 및 옵티마이저 초기화
denoiser = DistDiffusionBlock(d_model, num_heads, d_ff)
optimizer = optim.Adam(denoiser.parameters(), lr=0.001)

# 3. 더미 데이터 생성 (초기 분포)
mu_true = torch.randn(1, d_model) # (B, D)
var_true = torch.ones(1, d_model) * 0.1 # (B, D)
true_dist = GaussianDistribution(mu=mu_true, var=var_true)

# 4. 순방향 프로세스 (노이즈 추가)
betas = torch.linspace(0.0001, 0.02, T)
dist_t = true_dist
for t in range(T):
    dist_t = forward_step(dist_t, betas[t])

# 5. 역방향 프로세스 (디노이징 및 학습)
print("Starting training...")
for epoch in range(5):
    pred_dist = denoiser(dist_t, T) # t=T
    loss = loss_fn(pred_dist, true_dist, var_weight)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training finished.")

# 6. 샘플링
sampler = DiffusionSampler(denoiser, T)
initial_noise = GaussianDistribution(mu=torch.zeros(1, d_model), var=torch.ones(1, d_model))
final_sample = sampler.sample(initial_noise)

print("Final sample shape:", final_sample.shape)
