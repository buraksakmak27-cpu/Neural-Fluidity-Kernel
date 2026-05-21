# Neural Fluidity Kernel 🧠🌊

**An optimized deep learning infrastructure module for PyTorch that injects dynamic low-rank factorization (SVD) directly into the backward pass to eliminate catastrophic forgetting during continuous enterprise LLM fine-tuning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Deep Hook

In traditional deep learning, sequential training of Large Language Models (LLMs) on non-stationary data streams leads to **catastrophic forgetting**. When fine-tuning on new domain-specific alignment data, standard gradient descent updates modify the parameter weights uniformly across the entire dense parameter space. This causes **static parameter dislocation**—the erasure of orthogonal weight directions that encode historical knowledge.

**Neural Fluidity Kernel (NFK)** is high-performance AI infrastructure designed to solve this bottleneck. It replaces standard static projections with a dynamic, sparse-modulated low-rank layer where weight updates are constrained to a low-dimensional manifold. By fusing **Singular Value Decomposition (SVD)** and **top-k gating** directly into custom PyTorch autograd backward passes, NFK regularizes gradient flow. It selectively updates active low-rank projection spaces for new alignment tasks while shielding critical historical features from degradation.

---

## 2. Mathematical Foundation

### The Catastrophic Forgetting Problem in Sequential Learning
Consider a standard linear layer parameterised by $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. In a sequential learning paradigm, we optimize a loss function $\mathcal{L}$ over a task sequence $T_1, T_2, \dots, T_N$. The gradient update at step $t$ is given by:

$$W^{(t+1)} = W^{(t)} - \eta \nabla_{W} \mathcal{L}(W^{(t)})$$

Because $\nabla_{W} \mathcal{L}$ is typically dense and full-rank, the update vector spans the entire parameter space. This causes parameter drift along singular vectors that represent crucial features learned in preceding tasks. If the inputs for task $T_k$ ($k > 1$) reside in a subspace orthogonal to $T_1$, the network's mapping of the $T_1$ manifold is degraded, causing immediate performance degradation on past domains.

### Dynamic SVD Low-Rank Gradient Projection
NFK mitigates parameter drift by factorizing the weight matrix $W$ (or its dynamic update $\Delta W$) into a context-sensitive low-rank bottleneck:

$$W(z) = U \cdot \text{diag}\bigl(g(z)\bigr) \cdot V^{\top}$$

where:
* $V \in \mathbb{R}^{d_{\text{in}} \times d_{\text{latent}}}$ and $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{latent}}}$ represent the learned, orthogonal basis matrices.
* $d_{\text{latent}} \ll \min(d_{\text{in}}, d_{\text{out}})$ represents the maximum rank (or latent dimension).
* $z \in \mathbb{R}^{d_{\text{latent}}}$ is a dynamic, input-derived latent state vector.
* $g(z) \in \mathbb{R}^{d_{\text{latent}}}$ is a sparse gating function output.

By applying Singular Value Decomposition (SVD) on the initial weight matrix $W_0 \approx U_0 \Sigma_0 V_0^{\top}$ and updating only the low-rank components, the gradient update $\Delta W$ is mathematically restricted to:

$$\Delta W = U \Delta \Sigma V^{\top}$$

During the backward pass, the gradients are backpropagated through a custom autograd function. Let $\mathbf{x} \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ be the input tensor. The forward pass computes:

$$\mathbf{h} = \mathbf{x} V \quad \in \mathbb{R}^{B \times N \times d_{\text{latent}}}$$
$$\mathbf{h}_{\text{mod}} = \mathbf{h} \odot g(z) \quad \in \mathbb{R}^{B \times N \times d_{\text{latent}}}$$
$$\mathbf{y} = \mathbf{h}_{\text{mod}} U^{\top} + \mathbf{b} \quad \in \mathbb{R}^{B \times N \times d_{\text{out}}}$$

The gradient of the loss with respect to the low-rank bases $U$ and $V$ is calculated during the backward pass using outer products of the activations and the incoming gradient $\mathbf{g}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$:

$$\frac{\partial \mathcal{L}}{\partial U} = \mathbf{g}_y^{\top} \mathbf{h}_{\text{mod}} \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{latent}}}$$

$$\frac{\partial \mathcal{L}}{\partial V} = \mathbf{x}^{\top} \left( \left( \mathbf{g}_y U \right) \odot g(z) \right) \quad \in \mathbb{R}^{d_{\text{in}} \times d_{\text{latent}}}$$

### Top-K Singular Value Regularization
To prevent catastrophic interference, $g(z)$ enforces a hard top-$k$ sparsity constraint ($k \le d_{\text{latent}}$):

$$g(z) = \text{Top-K}\left(\text{Softmax}\left(W_g z\right), k\right)$$

This restricts the active singular vectors to a subset of size $k$ for each batch element. The gradient flow is regularized because parameters associated with inactive singular values receive zero gradient updates:

$$\frac{\partial \mathcal{L}}{\partial U_{i}} = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial V_{i}} = 0 \quad \forall i \notin \text{Top-K}(g(z))$$

This mathematical constraint preserves the historical orthogonal features in the inactive singular vectors while opening low-rank projection spaces in the active channels to absorb new alignment data.

---

## 3. Architecture & Features

Neural Fluidity Kernel is built with a decoupled, high-performance architecture:

```
                                  +-----------------------+
                                  |    PyTorch Frontend   |
                                  |  (FluidLinearLayer)   |
                                  +-----------+-----------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
        [CUDA Device Available]                               [CPU Fallback]
                       |                                             |
                       v                                             v
        +--------------+--------------+               +--------------+--------------+
        |     C++/CUDA JIT & AOT      |               |     PyTorch-Native Paths    |
        |    (fused forward kernel)   |               |     (autograd functions)    |
        +-----------------------------+               +-----------------------------+
```

* **FluidLinearLayer**: A drop-in, plug-and-play replacement for `torch.nn.Linear` that automatically handles initialization via CPU-bound SVD projections during loading or fine-tuning setup.
* **Hybrid JIT/AOT Backend**: Driven by `cuda_backend.py`, the system compiles custom CUDA kernels dynamically on-the-fly via `torch.utils.cpp_extension.load` if no pre-built binary is present. It switches to high-speed AOT bindings if pre-compiled via `setup.py`.
* **Zero-Divergence Native Fallback**: Seamless CPU execution path for non-CUDA platforms, ensuring code parity across local development workstations and multi-node GPU clusters.
* **Hugging Face Ecosystem Ready**: Structured to integrate with `transformers` models (e.g., Llama, Mistral) and `accelerate` configurations for deep-speed distributed scaling.

---

## 4. Installation & Local Development

Install the package in editable mode directly from your local repository to link dependencies and compile the CUDA kernel extension.

```bash
# Clone the repository
git clone https://github.com/buraksakmak27-cpu/Neural-Fluidity-Kernel.git
cd Neural-Fluidity-Kernel

# Install in editable mode
pip install -e .
```

The `setup.py` automatically detects your CUDA Toolkit and compiler toolchains to perform Ahead-of-Time (AOT) compilation if possible. If no CUDA driver is present, setup completes successfully in CPU-fallback mode, and JIT-compiles the CUDA bindings when a GPU becomes available.

---

## 5. Production Usage Example

The following script shows how to load a model, replace standard `nn.Linear` projections with `FluidLinearLayer` using Singular Value Decomposition, and run a training loop where the low-rank gradient updates protect historical parameters.

```python
import torch
import torch.nn as nn
from neural_fluidity import FluidLinearLayer

# 1. Define a standard PyTorch module
class TransformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Standard dense layers
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

# 2. Instantiate the model
d_model, d_ff = 4096, 4096
model = TransformerFeedForward(d_model, d_ff)

# 3. Swap standard nn.Linear layers with FluidLinearLayer
# latent_dim determines the maximum rank (r) of the SVD updates, top_k acts as the sparsity filter
model.w1 = FluidLinearLayer.from_linear(model.w1, latent_dim=128, top_k=32)
model.w2 = FluidLinearLayer.from_linear(model.w2, latent_dim=128, top_k=32)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Swapped layers successfully:")
print(model)

# 4. Standard training loop (SVD low-rank backward pass runs implicitly)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(5):
    # Dummy input sequence [Batch, SeqLen, HiddenDim]
    inputs = torch.randn(8, 128, d_model, device=device)
    targets = torch.randn(8, 128, d_model, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Under the hood, PyTorch routes gradients through our custom autograd function,
    # performing SVD projection and gradient updates along the low-rank manifold
    loss.backward()
    
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
```

---

## 6. Benchmark & Profiling Objectives

Our ongoing benchmarks focus on demonstrating parameters saving and gradient memory optimization at scale:

* **Llama-3 8B & Mistral 7B Architectures**: Quantifying catastrophic forgetting retention scores during sequential domain adaption (e.g., training on Code, Math, and Medical corpora sequentially).
* **VRAM Efficiency**: Evaluating activation memory overhead reductions during backpropagation by avoiding full-rank gradient updates on primary model projection paths.
* **Throughput Benchmarking**: Comparing fused CUDA kernel performance against PyTorch default linear matrices for hidden sizes $d_{\text{model}} \in \{4096, 8192, 16384\}$.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
