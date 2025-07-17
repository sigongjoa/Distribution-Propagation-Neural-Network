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
cell = DistributionCell(initial_distribution=init_dist)

# 2) 레이어/네트워크 구성
layer = DistributionLayer(cells=[cell for _ in range(10)])
net = DistributionNetwork(layers=[layer for _ in range(5)])

# 3) 전파
print("Starting propagation...")
# Initial propagation is now handled within the training loop
# net.forward(steps=10)
output = net.get_output()
print(f"Propagation finished. Output distributions from last layer: {len(output)}")

# Sample from one of the output distributions
if output:
    sample_output = output[0].sample()
    print(f"Sample from first output distribution: {sample_output}")
    log_prob_output = output[0].log_prob(sample_output)
    print(f"Log probability of the sampled value: {log_prob_output}")

    # Test reconstruct method
    reconstructed_value = cell.reconstruct()
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

# Create a network and optimizer
# Note: For a proper training loop, you'd typically re-initialize the network or reset its state
# For this demo, we'll just use the existing 'net' and 'optimizer'

optimizer = DistributionOptimizer(net, lr=0.01)
target_distributions = [GaussianDistribution(mu=torch.tensor(0.5, dtype=torch.float32), var=torch.tensor(0.8, dtype=torch.float32)) for _ in range(10)] # Example target

num_epochs = 5
print(f"\nStarting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # Forward pass
    net.zero_grad() # Clear gradients from previous step
    net.forward() # Propagate for one step
    output = net.get_output()

    # Calculate loss
    loss = optimizer.loss_fn(output, target_distributions)

    # Backward pass and optimize
    optimizer.step(loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete.")

# Final output after training
final_output = net.get_output()
print(f"\nFinal output distributions from last layer (first 3):")
for i, dist in enumerate(final_output[:3]):
    print(f"  Cell {i}: mu={dist.mu:.4f}, var={dist.var:.4f}")

print("\nExample run complete.")
