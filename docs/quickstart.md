# Quickstart

Welcome to DPNN (Distribution Propagation Neural Network)! This guide will help you get started with installing and using the library.

## Installation

You can install DPNN using pip:

```bash
pip install dpnn_lib
```

## Basic Usage

Here's a quick example demonstrating how to wrap a tensor into a Gaussian Distribution and pass it through a DPNN layer.

```python
import torch
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.layers.layer import DistLinear

# 1. Create a tensor
x = torch.randn(1, 10) # Example: batch_size=1, feature_dim=10

# 2. Wrap the tensor into a Gaussian Distribution
# For simplicity, let's assume initial variance is a small constant
initial_var = torch.ones_like(x) * 0.1
dist_x = GaussianDistribution(mu=x, var=initial_var)

print(f"Input Distribution Mu: {dist_x.mu.shape}")
print(f"Input Distribution Var: {dist_x.var.shape}")

# 3. Pass the distribution through a DPNN layer (e.g., DistLinear)
input_dim = 10
output_dim = 5
dpnn_layer = DistLinear(input_dim, output_dim)

dist_output = dpnn_layer(dist_x)

print(f"Output Distribution Mu: {dist_output.mu.shape}")
print(f"Output Distribution Var: {dist_output.var.shape}")

# You can then access the mean and variance of the output distribution
# For example, to get the log probability of a point:
# point = torch.randn(1, 5)
# log_prob = dist_output.log_prob(point)
# print(f"Log Probability of a random point: {log_prob}")

# Or combine two distributions:
# dist_y = GaussianDistribution(mu=torch.randn(1, 5), var=torch.ones(1, 5) * 0.1)
# combined_dist = dist_output.combine(dist_y)
# print(f"Combined Distribution Mu: {combined_dist.mu.shape}")
```

This is just a basic introduction. Refer to the `dpnn_lib` package documentation for more detailed usage and available modules.
