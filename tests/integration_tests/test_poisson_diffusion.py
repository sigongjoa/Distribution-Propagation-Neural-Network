# test_poisson_diffusion.py

import torch
from dpnn_lib.diffusion.poisson_diffusion import PoissonDiffusion
from dpnn_lib.models.poisson_denoiser import PoissonDenoiser
from dpnn_lib.distributions.poisson import PoissonDistribution

# 간단히 forward/reverse 동작 확인
hidden_dim = 32
betas = torch.linspace(0.001, 0.02, steps=100)
timesteps = len(betas)
prior_rate = torch.tensor(3.0)

denoiser = PoissonDenoiser(hidden_dim)
diffusion = PoissonDiffusion(denoiser, betas, prior_rate, timesteps)

# 초기 분포
x0 = PoissonDistribution(torch.tensor(3.0))
xs = diffusion.run_forward(x0)
xt = xs[-1] # Get the last noisy distribution
x0_rec = diffusion.run_reverse(xt)
print("Original rate:", x0.lambda_param.item(), "Reconstructed rate:", x0_rec.lambda_param.item())
