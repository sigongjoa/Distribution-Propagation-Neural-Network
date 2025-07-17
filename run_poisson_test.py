import sys
import os

# Add the parent directory of dpnn_lib to the Python path
sys.path.insert(0, os.path.abspath('/mnt/d/progress/DPNN'))

import torch
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.distributions.poisson import PoissonDistribution
from dpnn_lib.distributions.dirichlet import DirichletDistribution
from dpnn_lib.cells.cell import DistributionCell
from dpnn_lib.layers.layer import DistributionLayer
from dpnn_lib.layers.network import DistributionNetwork
from dpnn_lib.utils.optim import DistributionOptimizer

# 1) 분포 생성
init_dist = GaussianDistribution(mu=torch.tensor(0.0, dtype=torch.float32), var=torch.tensor(1.0, dtype=torch.float32))
# Create a list of cells, each with its initial distribution
cells_for_layer = [DistributionCell(initial_distribution=init_dist) for _ in range(10)]

# 2) 레이어/네트워크 구성
layer = DistributionLayer(cells=cells_for_layer)
# Create a list of layers, each with its cells
layers_for_network = [layer for _ in range(5)]
net = DistributionNetwork(layers=layers_for_network)

# The initial distributions for the network's first forward pass
initial_distributions_for_network = []
for cell in cells_for_layer:
    # Detach the mu and var tensors to break the graph for initial distributions
    detached_mu = cell.initial_distribution.mu.detach().requires_grad_(True)
    detached_var = cell.initial_distribution.var.detach().requires_grad_(True)
    initial_distributions_for_network.append(GaussianDistribution(mu=detached_mu, var=detached_var))

# 3) 전파 (Initial output before training loop)
output_before_training = net.forward(initial_distributions_for_network) # Use forward to get initial output
print("Starting propagation...")
print(f"Propagation finished. Output distributions from last layer: {len(output_before_training)}")

# Sample from one of the output distributions
if output_before_training:
    sample_output = output_before_training[0].sample()
    print(f"Sample from first output distribution: {sample_output}")
    log_prob_output = output_before_training[0].log_prob(sample_output)
    print(f"Log probability of the sampled value: {log_prob_output}")

    # Test reconstruct method
    reconstructed_value = cells_for_layer[0].reconstruct(cells_for_layer[0].initial_distribution)
    print(f"Reconstructed value from initial cell: {reconstructed_value}")

# Test PoissonDistribution
print("\nTesting PoissonDistribution:")
poisson_dist = PoissonDistribution(lambda_param=3.0)
poisson_sample = poisson_dist.sample()
print(f"Poisson sample (lambda=3.0): {poisson_sample}")
poisson_log_prob = poisson_dist.log_prob(poisson_sample)
print(f"Poisson log_prob of sample: {poisson_log_prob}")

# Test DirichletDistribution
print("\nTesting DirichletDistribution:")
dirichlet_dist = DirichletDistribution(alpha_vector=[1.0, 2.0, 3.0])
dirichlet_sample = dirichlet_dist.sample()
print(f"Dirichlet sample (alpha=[1.0, 2.0, 3.0]): {dirichlet_sample}")
# Ensure the sum is close to 1 for log_prob
dirichlet_sample_normalized = dirichlet_sample / sum(dirichlet_sample)
dirichlet_log_prob = dirichlet_dist.log_prob(dirichlet_sample_normalized)
print(f"Dirichlet log_prob of normalized sample: {dirichlet_log_prob}")

# 4) 최적화
print("\nTesting Optimization:")

optimizer = DistributionOptimizer(net, lr=0.01)
target_distributions = [GaussianDistribution(mu=torch.tensor(0.5, dtype=torch.float32), var=torch.tensor(0.8, dtype=torch.float32)) for _ in range(10)] # Example target

num_epochs = 5
print(f"\nStarting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # Forward pass
    # Pass the initial distributions for each epoch
    output = net.forward(initial_distributions_for_network) # Pass initial distributions

    # Calculate loss
    loss = optimizer.loss_fn(output, target_distributions)

    # Backward pass
    loss.backward() # Call backward here

    # Optimize
    optimizer.step() # No longer pass 'loss' argument

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete.")

# Final output after training
final_output = net.forward(initial_distributions_for_network) # Get final output after training
print(f"\nFinal output distributions from last layer (first 3):")
for i, dist in enumerate(final_output[:3]):
    print(f"  Cell {i}: mu={dist.mu:.4f}, var={dist.var:.4f}")

print("\nExample run complete.")
