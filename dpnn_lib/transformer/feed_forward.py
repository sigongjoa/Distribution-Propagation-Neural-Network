
import torch.nn as nn
import torch.nn.functional as F
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1_mu   = nn.Linear(d_model, d_ff)
        self.fc1_var  = nn.Linear(d_model, d_ff)
        self.fc2_mu   = nn.Linear(d_ff, d_model)
        self.fc2_var  = nn.Linear(d_ff, d_model)

    def forward(self, dist):
        m1 = F.gelu(self.fc1_mu(dist.mu))
        v1 = F.softplus(self.fc1_var(dist.var))
        m2 = self.fc2_mu(m1)
        v2 = F.softplus(self.fc2_var(v1))
        return GaussianDistribution(mu=m2, var=v2)
