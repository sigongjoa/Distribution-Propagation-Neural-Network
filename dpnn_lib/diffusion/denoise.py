
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.transformer.transformer_block import DistTransformerBlock

class DistDiffusionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.transformer_block = DistTransformerBlock(d_model, num_heads, d_ff)

    def forward(self, dist_t: GaussianDistribution, t: int):
        # The time step `t` is not used in this implementation, but is kept for API consistency.
        return self.transformer_block(dist_t)
