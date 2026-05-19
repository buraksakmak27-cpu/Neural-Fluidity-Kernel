import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernel import FluidLinear, LatentStateUpdater

class FluidAttention(nn.Module):
    """
    A Multi-Head Self-Attention block using FluidLinear projections.
    Features:
      - Uses a shared LatentStateUpdater to update/create the latent state.
      - Uses FluidLinear for Query, Key, Value, and Output projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, latent_dim: int, top_k: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        self.top_k = top_k
        
        # Latent state updater
        self.latent_updater = LatentStateUpdater(embed_dim, latent_dim)
        
        # Projections
        self.q_proj = FluidLinear(embed_dim, embed_dim, latent_dim, top_k)
        self.k_proj = FluidLinear(embed_dim, embed_dim, latent_dim, top_k)
        self.v_proj = FluidLinear(embed_dim, embed_dim, latent_dim, top_k)
        self.out_proj = FluidLinear(embed_dim, embed_dim, latent_dim, top_k)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, z_prev: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        x: Input tensor of shape [B, N, embed_dim]
        z_prev: Previous latent state of shape [B, latent_dim] (optional)
        attn_mask: Attention mask (optional)
        """
        B, N, C = x.shape
        
        # 1. Update/generate the latent state z
        z = self.latent_updater(x, z_prev) # [B, latent_dim]
        
        # 2. Project Q, K, V using FluidLinear with the dynamic latent state z
        q = self.q_proj(x, z)  # [B, N, embed_dim]
        k = self.k_proj(x, z)  # [B, N, embed_dim]
        v = self.v_proj(x, z)  # [B, N, embed_dim]
        
        # 3. Reshape Q, K, V for multi-head attention
        # Shape: [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Scaled dot-product attention
        # scores: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # context: [B, num_heads, N, head_dim]
        context = torch.matmul(attn_probs, v)
        
        # Reshape context back to [B, N, embed_dim]
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        
        # 5. Output projection using FluidLinear
        output = self.out_proj(context, z)
        
        return output, z, attn_probs
