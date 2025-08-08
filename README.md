# Distribution Propagation Neural Network (DPNN)

This repository contains a Proof of Concept (PoC) implementation of a Distribution Propagation Neural Network (DPNN). The core idea behind DPNN is to represent and propagate information as probability distributions (e.g., Gaussian) rather than fixed point values, allowing the network to inherently model uncertainty. This approach enables the network to not only process data but also to quantify and propagate the uncertainty associated with that data, leading to more robust and interpretable models.

## Key Features

-   **Distributional Core:** Implements various probability distributions (Gaussian, Dirichlet, Poisson) as first-class citizens, allowing operations directly on their parameters (mean, variance, etc.).
-   **Distribution-based Transformer:** A transformer architecture where token embeddings and internal representations are handled as distributions, enabling uncertainty propagation through self-attention and feed-forward layers.
-   **Distribution-based Diffusion Models:** A novel approach to diffusion models where the forward and reverse processes operate on distributions, allowing for probabilistic denoising and generation.
-   **Graph Operations on Distributions:** Explores how distributions can propagate and interact on graph structures, including Graph PDE (Partial Differential Equation) layers and Neural ODE (Ordinary Differential Equation) functions that operate on distributional node states.

## Installation

To set up the environment and run the examples, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sigongjoa/Distribution-Propagation-Neural-Network.git
    cd Distribution-Propagation-Neural-Network
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install required packages:**
    ```bash
    pip install torch torch_geometric torchdiffeq matplotlib
    ```
    *(Note: Ensure you have the correct PyTorch version compatible with your CUDA setup if you plan to use a GPU. Refer to the official PyTorch documentation for specific installation instructions.)*

## Usage Examples

This repository includes several example scripts demonstrating the different components of the DPNN.

To run an example, activate your virtual environment and execute the script as a Python module from the project root directory:

-   **Transformer Demo:**
    ```bash
    python -m dpnn_lib.examples.transformer_demo
    ```

-   **Diffusion Demo:**
    ```bash
    python -m dpnn_lib.examples.diffusion_demo
    ```

-   **Graph PDE/ODE Demo:**
    ```bash
    python -m dpnn_lib.examples.graph_pde_ode_demo
    ```

-   **Graph Transformer PoC Test:**
    ```bash
    python -m graph_transformer_poc_test
    ```

## Related Work and Context

DPNN builds upon the rich history of neural networks and probabilistic modeling. While traditional neural networks often output point estimates, DPNN explicitly models and propagates uncertainty through the network layers. This differentiates it from approaches that only quantify uncertainty at the output layer or rely on Monte Carlo sampling. Our work shares conceptual similarities with research in Bayesian Deep Learning and uncertainty quantification in AI, but focuses on a novel architectural paradigm where distributions are the fundamental units of information. For instance, unlike some DPGN (Deep Probabilistic Graph Networks) lines that focus on specific graph structures or few-shot learning, DPNN aims for a more general framework of distribution propagation across various neural network architectures.

## Project Status and Meta

This project is currently a Proof of Concept. We are actively working on:

-   **PyPI Distribution:** Planning for official PyPI package distribution to simplify installation.
-   **Version Tagging & Changelog:** Implementing proper versioning and maintaining a detailed changelog for releases.
-   **Continuous Integration (CI):** Setting up comprehensive CI pipelines for automated documentation builds, unit tests, and integration tests to ensure code quality and reliability. You can view the status of our documentation build [here](https://sigongjoa.github.io/Distribution-Propagation-Neural-Network/).

## Project Structure (Proposed)

```
dpnn_lib/
├── distributions/                # Information Unit: Distributions
│   ├── core/                     # Pure Distribution Logic
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dirichlet.py
│   │   ├── gaussian.py
│   │   └── poisson.py
│   ├── transformer/              # Distribution-based Transformer Implementation
│   │   ├── __init__.py
│   │   ├── attention.py          # Q/K/V → Distribution Parameters
│   │   ├── feed_forward.py       # DistributionCell etc.
│   │   └── transformer.py        # DistTransformer Layer/Model
│   └── diffusion/                # Distribution-based Diffusion Implementation
│       ├── __init__.py
│       ├── forward.py            # Noise Addition
│       ├── denoise.py            # Distribution-Cell based Denoising
│       ├── sampler.py            # Distribution Sampling Logic
│       └── loss.py               # Loss Function for Distribution Parameters
├── graph_ops/                    # Graph Operations on Distributions (PDE/ODE, graph-attention)
│   ├── __init__.py
│   ├── pde_layer.py
│   ├── ode_func.py
│   ├── graph_dist_attention.py
│   └── graph_transformer_block.py
├── models/                       # (Optional) Existing non-distribution based standard models
│   ├── transformer/
│   └── diffusion/
└── utils/                        # Common Utilities
    ├── __init__.py
    ├── metrics.py
    └── optim.py
```

*(Note: The `models/` directory is currently optional and might be removed if not used for non-distributional models. The proposed structure aims for clear separation of concerns, where `distributions/` contains all logic directly related to operating on and transforming distributions, while `graph_ops/` provides fundamental graph-based operations on these distributions.)*