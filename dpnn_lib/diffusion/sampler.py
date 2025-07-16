
import torch
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.diffusion.denoise import DistDiffusionBlock

class DiffusionSampler:
    def __init__(self, denoise_block: DistDiffusionBlock, T: int):
        self.denoise_block = denoise_block
        self.T = T

    def sample(self, initial_dist: GaussianDistribution):
        dist_t = initial_dist
        for t in reversed(range(1, self.T + 1)):
            dist_t = self.denoise_block(dist_t, t)
            x_sampled = dist_t.sample()
            dist_t = GaussianDistribution(mu=x_sampled, var=dist_t.var)
        return dist_t.sample()
