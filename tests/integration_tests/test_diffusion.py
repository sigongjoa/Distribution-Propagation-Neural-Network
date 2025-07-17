import torch
import torch.optim as optim
from ...dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.diffusion.gaussian_diffusion import GaussianDiffusion, DistDiffusionBlock # Import new classes

# 1. 하이퍼파라미터 설정
d_model = 256      # 모델 차원
num_heads = 4      # 어텐션 헤드 수
d_ff = 1024        # 피드포워드 신경망의 내부 차원
T = 100            # 총 타임스텝 수
var_weight = 1.0   # 분산 손실 가중치

# 2. 모델 및 옵티마이저 초기화
denoiser = DistDiffusionBlock(d_model, num_heads, d_ff)
betas = torch.linspace(0.0001, 0.02, T)
diffusion_process = GaussianDiffusion(denoiser, T, betas) # Use GaussianDiffusion
optimizer = optim.Adam(denoiser.parameters(), lr=0.001)

# 3. 더미 데이터 생성 (초기 분포)
mu_true = torch.randn(1, d_model) # (B, D)
var_true = torch.ones(1, d_model) * 0.1 # (B, D)
true_dist = GaussianDistribution(mu=mu_true, var=var_true)

# 4. 순방향 프로세스 (노이즈 추가)
# Use run_forward from GaussianDiffusion
dist_t_trajectory = diffusion_process.run_forward(true_dist)
dist_t = dist_t_trajectory[-1] # Get the final noisy distribution

# 5. 역방향 프로세스 (디노이징 및 학습)
print("Starting training...")
for epoch in range(5):
    # Pick a random timestep for training
    t_train = torch.randint(0, T, (1,)).item() + 1 # Random timestep from 1 to T
    
    # Get the noisy distribution at t_train
    noisy_dist_at_t = dist_t_trajectory[t_train]

    # Denoise and calculate loss
    pred_dist = diffusion_process.denoise_block(noisy_dist_at_t, t_train)
    loss = diffusion_process.loss(true_dist, noisy_dist_at_t, pred_dist, t_train) # Pass all required args
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training finished.")

# 6. 샘플링
# Use run_reverse from GaussianDiffusion
initial_noise = GaussianDistribution(mu=torch.zeros(1, d_model), var=torch.ones(1, d_model))
final_sample_dist = diffusion_process.run_reverse(initial_noise)
final_sample = final_sample_dist.sample()

print("Final sample shape:", final_sample.shape)