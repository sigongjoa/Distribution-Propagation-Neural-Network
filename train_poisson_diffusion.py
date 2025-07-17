# train_poisson_diffusion.py

import torch
from torch.utils.data import DataLoader
from dpnn_lib.distributions.poisson import PoissonDistribution
from dpnn_lib.diffusion.poisson_diffusion import PoissonDiffusion
from dpnn_lib.models.poisson_denoiser import PoissonDenoiser

# 하이퍼파라미터
betas = torch.linspace(0.001, 0.02, steps=100)
prior_rate = torch.tensor(3.0)
timesteps = len(betas)
hidden_dim = 32
epochs = 10
batch_size = 64

# 모델·확산 클래스 초기화
denoiser = PoissonDenoiser(hidden_dim)
diffusion = PoissonDiffusion(denoiser, betas, prior_rate, timesteps)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

# 더미 데이터셋: 초기 λ0 샘플링
class LambdaDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.rates = torch.rand(size) * 5  # [0,5) 구간

    def __len__(self):
        return len(self.rates)
    def __getitem__(self, idx):
        return PoissonDistribution(self.rates[idx])

def poisson_collate_fn(batch):
    # batch는 PoissonDistribution 객체의 리스트
    # 각 PoissonDistribution 객체에서 lambda_param을 추출하여 텐서로 묶음
    lambda_params = [item.lambda_param for item in batch]
    return PoissonDistribution(torch.stack(lambda_params))

dataloader = DataLoader(LambdaDataset(10000), batch_size=batch_size, shuffle=True, collate_fn=poisson_collate_fn)

# 학습 루프
print("Starting Poisson Diffusion Training...")
for epoch in range(epochs):
    total_loss = 0.
    for x0_dist in dataloader:
        # 1) forward diffusion
        xs = diffusion.run_forward(x0_dist) # x0_dist는 PoissonDistribution 객체
        
        # 2) 랜덤 t 선택
        t = torch.randint(0, timesteps, (batch_size,))
        
        # 배치별 xt (xs는 리스트이므로, 각 요소에서 해당 인덱스의 분포를 가져와야 함)
        # xs는 List[PoissonDistribution]이므로, 각 PoissonDistribution에서 lambda_param을 추출
        # 그리고 t에 해당하는 xt를 선택
        xt_list = []
        for i in range(batch_size):
            # xs[t[i]]는 PoissonDistribution 객체
            xt_list.append(xs[t[i]].lambda_param)
        xt_tensor = torch.stack(xt_list) # (batch_size,)
        xt_dist = PoissonDistribution(xt_tensor) # PoissonDistribution 객체로 다시 래핑

        # 3) denoise
        x0_hat = diffusion.denoise_block(xt_dist)
        
        # 4) loss 계산 및 역전파
        loss = diffusion.loss(x0_dist, xt_dist, x0_hat, t) # x0_dist는 배치 크기에 맞게 전달
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
print("Poisson Diffusion Training Finished.")
