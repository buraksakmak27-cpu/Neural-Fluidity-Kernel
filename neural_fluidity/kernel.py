import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_backend import get_cuda_backend

class FluidLinear(nn.Module):
    """
    A Dynamic Sparse-Modulated Linear Layer replacing nn.Linear.
    Uses low-rank factorization (U @ diag(g(z)) @ V.T) and slices bases
    dynamically using Top-K gating on a latent state 'z'.
    """
    def __init__(self, in_features: int, out_features: int, latent_dim: int, top_k: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.top_k = min(top_k, latent_dim)
        
        # Static low-rank bases
        # Replaces dense matrix of size [out_features, in_features]
        # with low-rank factorized parameters [in_features, latent_dim] and [out_features, latent_dim]
        # Initialized with scaled normal to keep variance stable
        self.V = nn.Parameter(torch.randn(in_features, latent_dim) * (2.0 / (in_features + latent_dim)) ** 0.5)
        self.U = nn.Parameter(torch.randn(out_features, latent_dim) * (2.0 / (out_features + latent_dim)) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gating network to produce modulation coefficients from latent state z
        self.gate_proj = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        x: Input tensor of shape [B, N, in_features] or [B, in_features]
        z: Latent state tensor of shape [B, latent_dim]
        """
        # Handle 2D input [B, in_features] -> unsqueeze to [B, 1, in_features]
        is_2d = False
        if x.dim() == 2:
            is_2d = True
            x = x.unsqueeze(1)
            
        B, N, C = x.shape
        
        # 1. Compute gating scores from latent state
        gate_scores = self.gate_proj(z)  # [B, latent_dim]
        gate_probs = F.softmax(gate_scores, dim=-1) # [B, latent_dim]
        
        # 2. Top-K Sparsification
        topk_vals, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Re-normalize Top-K values to maintain scale
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)
        
        # --- Custom CUDA Fused Kernel Execution ---
        cuda_backend = get_cuda_backend()
        if cuda_backend is not None and x.is_cuda:
            y = cuda_backend.forward(x, self.V, self.U, self.bias, topk_indices, topk_vals)
            if is_2d:
                y = y.squeeze(1)
            return y
            
        # --- PyTorch-Native Fallback Mode ---
        # 3. Slice the V and U parameter matrices dynamically per batch element

        # V: [in_features, latent_dim] -> V_expanded: [B, in_features, latent_dim]
        V_expanded = self.V.unsqueeze(0).expand(B, -1, -1)
        idx_V = topk_indices.unsqueeze(1).expand(-1, self.in_features, -1) # [B, in_features, K]
        V_sliced = torch.gather(V_expanded, 2, idx_V) # [B, in_features, K]
        
        # U: [out_features, latent_dim] -> U_expanded: [B, out_features, latent_dim]
        U_expanded = self.U.unsqueeze(0).expand(B, -1, -1)
        idx_U = topk_indices.unsqueeze(1).expand(-1, self.out_features, -1) # [B, out_features, K]
        U_sliced = torch.gather(U_expanded, 2, idx_U) # [B, out_features, K]
        
        # 4. Computation using batched matrix multiplications
        # H = X * V_sliced -> [B, N, K]
        h = torch.bmm(x, V_sliced)
        
        # Modulate low-rank representation: H_mod = H * topk_vals
        # topk_vals is [B, K] -> unsqueeze to [B, 1, K] for broadcasting
        h_mod = h * topk_vals.unsqueeze(1)
        
        # Y = H_mod * U_sliced^T -> [B, N, out_features]
        y = torch.bmm(h_mod, U_sliced.transpose(1, 2))
        
        # Add bias
        y = y + self.bias
        
        if is_2d:
            y = y.squeeze(1)
            
        return y


class LatentStateUpdater(nn.Module):
    """
    Dynamically computes and updates the low-dimensional latent state z
    from input sequence x.
    """
    def __init__(self, in_features: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features, latent_dim)
        
    def forward(self, x: torch.Tensor, z_prev: torch.Tensor = None):
        """
        x: Input tensor [B, N, in_features] or [B, in_features]
        z_prev: Previous latent state [B, latent_dim]
        """
        if x.dim() == 2:
            x_summary = x
        else:
            # Mean pool across sequence dimension
            x_summary = x.mean(dim=1) # [B, in_features]
            
        z_new = self.proj(x_summary) # [B, latent_dim]
        
        if z_prev is not None:
            # Simple recurrent update with residual connection
            z_new = 0.9 * z_prev + 0.1 * torch.tanh(z_new)
        else:
            z_new = torch.tanh(z_new)
            
        return z_new
