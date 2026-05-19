import torch
import torch.nn as nn
import time
from neural_fluidity.tracker import MemoryTracker
from neural_fluidity.kernel import FluidLinear
from neural_fluidity.attention import FluidAttention

def run_linear_benchmark(device, batch_size=32, seq_len=128, in_dim=4096, out_dim=4096, latent_dim=128):
    print("=" * 80)
    print(f"RUNNING LINEAR LAYER BENCHMARK ON {device.type.upper()}")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | In Dim: {in_dim} | Out Dim: {out_dim} | Latent Dim: {latent_dim}")
    print("=" * 80)
    
    # Generate dummy input
    x = torch.randn(batch_size, seq_len, in_dim, device=device, requires_grad=True)
    z = torch.randn(batch_size, latent_dim, device=device) # Latent state
    
    # --- 1. Standard nn.Linear Baseline ---
    standard_linear = nn.Linear(in_dim, out_dim).to(device)
    with MemoryTracker("Standard nn.Linear (Forward + Backward)") as tracker:
        # Forward pass
        y_std = standard_linear(x)
        loss_std = y_std.sum()
        # Backward pass
        loss_std.backward()
    
    # Keep standard results
    std_cuda = tracker.results.get("cuda_peak", 0)
    
    # Free memory
    del standard_linear, y_std, loss_std
    torch.cuda.empty_cache()
    
    # --- 2. FluidLinear with varying top_k ---
    top_k_options = [4, 16, 64, 128]
    fluid_results = {}
    
    for k in top_k_options:
        fluid_linear = FluidLinear(in_dim, out_dim, latent_dim, top_k=k).to(device)
        with MemoryTracker(f"FluidLinear (top_k={k}) (Forward + Backward)") as tracker:
            y_fluid = fluid_linear(x, z)
            loss_fluid = y_fluid.sum()
            loss_fluid.backward()
        
        fluid_results[k] = tracker.results.get("cuda_peak", 0)
        
        # Free memory
        del fluid_linear, y_fluid, loss_fluid
        torch.cuda.empty_cache()
        
    print("\n--- Summary: Linear Layer VRAM Peak Consumption (MB) ---")
    print(f"Standard nn.Linear: {std_cuda:.2f} MB")
    for k, vram in fluid_results.items():
        reduction = ((std_cuda - vram) / std_cuda * 100) if std_cuda > 0 else 0
        print(f"FluidLinear (top_k={k:3d}): {vram:.2f} MB (VRAM reduction: {reduction:.1f}%)")


def run_attention_benchmark(device, batch_size=16, seq_len=256, embed_dim=2048, num_heads=8, latent_dim=128):
    print("\n" + "=" * 80)
    print(f"RUNNING ATTENTION BLOCK BENCHMARK ON {device.type.upper()}")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | Embed Dim: {embed_dim} | Heads: {num_heads} | Latent Dim: {latent_dim}")
    print("=" * 80)
    
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    
    # --- 1. Standard nn.MultiheadAttention Baseline ---
    standard_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
    # Target projection parameters (Q, K, V, Out) in standard attention are huge.
    with MemoryTracker("Standard MultiheadAttention (Forward + Backward)") as tracker:
        y_std, _ = standard_attn(x, x, x)
        loss_std = y_std.sum()
        loss_std.backward()
        
    std_cuda = tracker.results.get("cuda_peak", 0)
    
    del standard_attn, y_std, loss_std
    torch.cuda.empty_cache()
    
    # --- 2. FluidAttention with varying top_k ---
    top_k_options = [4, 16, 64, 128]
    fluid_results = {}
    
    for k in top_k_options:
        fluid_attn = FluidAttention(embed_dim, num_heads, latent_dim, top_k=k).to(device)
        with MemoryTracker(f"FluidAttention (top_k={k}) (Forward + Backward)") as tracker:
            y_fluid, _, _ = fluid_attn(x)
            loss_fluid = y_fluid.sum()
            loss_fluid.backward()
            
        fluid_results[k] = tracker.results.get("cuda_peak", 0)
        
        del fluid_attn, y_fluid, loss_fluid
        torch.cuda.empty_cache()
        
    print("\n--- Summary: Attention Block VRAM Peak Consumption (MB) ---")
    print(f"Standard Attention: {std_cuda:.2f} MB")
    for k, vram in fluid_results.items():
        reduction = ((std_cuda - vram) / std_cuda * 100) if std_cuda > 0 else 0
        print(f"FluidAttention (top_k={k:3d}): {vram:.2f} MB (VRAM reduction: {reduction:.1f}%)")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup CUDA if available
    if device.type == "cuda":
        print("Warming up CUDA...")
        dummy = torch.randn(100, 100, device=device)
        dummy = dummy @ dummy
        del dummy
        torch.cuda.empty_cache()
        
    run_linear_benchmark(device)
    run_attention_benchmark(device)
