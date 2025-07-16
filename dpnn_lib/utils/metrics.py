import torch
import torch.nn.functional as F
from typing import List
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.distributions.gaussian import GaussianDistribution

class Metrics:
    @staticmethod
    def kl_divergence(p: BaseDistribution, q: BaseDistribution):
        # Implement KL divergence calculation
        from dpnn_lib.distributions.gaussian import GaussianDistribution

        if not isinstance(p, GaussianDistribution) or not isinstance(q, GaussianDistribution):
            raise TypeError("KL divergence for now only supported for GaussianDistribution.")

        # KL divergence between two Gaussian distributions N(mu_p, var_p) and N(mu_q, var_q)
        # KL(P || Q) = 0.5 * (log(var_q / var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1)
        if p.var <= 0 or q.var <= 0:
            raise ValueError("Variances must be positive for KL divergence calculation.")

        # Convert to torch tensors for autograd
        mu_p, var_p = p.mu, p.var
        mu_q, var_q = q.mu, q.var

        kl_div = 0.5 * (torch.log(var_q / var_p) + (var_p + (mu_p - mu_q)**2) / var_q - 1)
        return kl_div

    @staticmethod
    def perplexity(logits, labels):
        cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return torch.exp(cross_entropy)

    @staticmethod
    def accuracy(logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == labels).float().mean()

    @staticmethod
    def visualize_distribution(distribution: BaseDistribution):
        # Implement distribution visualization (e.g., using matplotlib)
        pass
