# test_dirichlet_diffusion.py

import torch
from dpnn_lib.diffusion.dirichlet_diffusion import DirichletDiffusion
from dpnn_lib.models.dirichlet_denoiser import DirichletDenoiser
from dpnn_lib.distributions.dirichlet import DirichletDistribution

# 간단히 forward/reverse 동작 확인
hidden_dim = 32
betas = torch.linspace(0.001, 0.02, steps=100)
timesteps = len(betas)
prior_alpha = torch.tensor([1.0, 1.0, 1.0])
alpha_dim = len(prior_alpha)

denoiser = DirichletDenoiser(alpha_dim, hidden_dim)
diffusion = DirichletDiffusion(denoiser, betas, prior_alpha, timesteps)

# 초기 분포
x0 = DirichletDistribution(torch.tensor([3.0, 2.0, 1.0]))
xs = diffusion.run_forward(x0)
xt = xs[-1] # Get the last noisy distribution
x0_rec = diffusion.run_reverse(xt)
print("Original alpha:", x0.alpha_vector.tolist(), "Reconstructed alpha:", x0_rec.alpha_vector.tolist())
