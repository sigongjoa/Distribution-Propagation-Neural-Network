# train_dirichlet_diffusion.py

import torch
from torch.utils.data import DataLoader
from dpnn_lib.distributions.dirichlet import DirichletDistribution
from dpnn_lib.diffusion.dirichlet_diffusion import DirichletDiffusion
from dpnn_lib.models.dirichlet_denoiser import DirichletDenoiser

# 하이퍼파라미터
betas = torch.linspace(0.001, 0.02, steps=100)
prior_alpha = torch.tensor([1.0, 1.0, 1.0]) # 3차원 Dirichlet
timesteps = len(betas)
hidden_dim = 32
epochs = 10
batch_size = 64
alpha_dim = len(prior_alpha)

# 모델·확산 클래스 초기화
denoiser = DirichletDenoiser(alpha_dim, hidden_dim, timesteps=timesteps, time_embed_dim=1)
diffusion = DirichletDiffusion(denoiser, betas, prior_alpha, timesteps)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

# 더미 데이터셋: 초기 α0 샘플링
class AlphaDataset(torch.utils.data.Dataset):
    def __init__(self, size, alpha_dim):
        self.alphas = torch.rand(size, alpha_dim) * 5 + 0.1 # [0.1, 5.1) 구간

    def __len__(self):
        return len(self.alphas)
    def __getitem__(self, idx):
        return DirichletDistribution(self.alphas[idx])

def dirichlet_collate_fn(batch):
    alpha_vectors = [item.alpha_vector for item in batch]
    return DirichletDistribution(torch.stack(alpha_vectors))

dataloader = DataLoader(AlphaDataset(10000, alpha_dim), batch_size=batch_size, shuffle=True, collate_fn=dirichlet_collate_fn)

# 학습 루프
print("Starting Dirichlet Diffusion Training...")
for epoch in range(epochs):
    total_loss = 0.
    for x0_dist in dataloader:
        # 1) forward diffusion
        xs = diffusion.run_forward(x0_dist) # x0_dist는 DirichletDistribution 객체
        
        # 2) 랜덤 t 선택
        t = torch.randint(0, timesteps, (batch_size,))
        
        # 배치별 xt (xs는 리스트이므로, 각 요소에서 해당 인덱스의 분포를 가져와야 함)
        xt_list = []
        for i in range(batch_size):
            xt_list.append(xs[t[i]].alpha_vector[i])
        xt_tensor = torch.stack(xt_list) # (batch_size, alpha_dim)
        xt_dist = DirichletDistribution(xt_tensor) # DirichletDistribution 객체로 다시 래핑

        # 3) denoise
        x0_hat = diffusion.denoise_block(xt_dist, t) # t는 배치 크기에 맞게 전달
        
        # 4) loss 계산 및 역전파
        loss = diffusion.loss(x0_dist, xt_dist, x0_hat, t) # x0_dist는 배치 크기에 맞게 전달
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
print("Dirichlet Diffusion Training Finished.")
