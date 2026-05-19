import torch
import torch.nn as nn
import time
from neural_fluidity.tracker import MemoryTracker
from neural_fluidity.kernel import FluidLinear
from neural_fluidity.cuda_backend import get_cuda_backend

def run_cuda_validation_and_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dimensions and Iterations (Adaptive for CPU / CUDA)
    if device.type == "cuda":
        batch_size = 32
        seq_len = 128
        num_iters = 100
    else:
        batch_size = 2
        seq_len = 16
        num_iters = 5
        
    in_dim = 8192
    out_dim = 8192
    latent_dim = 128
    top_k = 16
    
    print("=" * 80)
    print(f"CUDA BENCHMARK RUNNER (Hidden Dim = 8192)")
    print(f"Current Device: {device.type.upper()}")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | Iterations: {num_iters}")
    print("=" * 80)
    
    if device.type != "cuda":
        print("[WARNING] CUDA device is not active on this environment.")
        print("To test the actual custom CUDA speedups, please run this on a GPU-enabled platform.")
        print("Falling back to Native PyTorch CPU execution...")
        print("=" * 80)
        
    # 2. Inputs
    x = torch.randn(batch_size, seq_len, in_dim, device=device, requires_grad=False)
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # 3. Instantiate Layers
    fluid_layer = FluidLinear(in_dim, out_dim, latent_dim, top_k=top_k).to(device)
    std_linear = nn.Linear(in_dim, out_dim).to(device)
    
    # 4. Measure Standard nn.Linear
    print("Benchmarking Standard nn.Linear...")
    # Warmup
    for _ in range(3):
        _ = std_linear(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = std_linear(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    std_avg_latency = (time.perf_counter() - start_time) * 1000 / num_iters
    
    # VRAM check
    with MemoryTracker("Standard nn.Linear VRAM") as tracker:
        _ = std_linear(x)
    std_vram = tracker.results.get("cuda_peak", 0.0)
    
    # 5. Measure FluidLinear PyTorch-Native
    import neural_fluidity.kernel as kernel_module
    orig_get_backend = kernel_module.get_cuda_backend
    kernel_module.get_cuda_backend = lambda: None
    
    print("Benchmarking FluidLinear PyTorch-Native...")
    # Warmup
    for _ in range(3):
        _ = fluid_layer(x, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = fluid_layer(x, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    native_avg_latency = (time.perf_counter() - start_time) * 1000 / num_iters
    
    # VRAM check
    with MemoryTracker("FluidLinear Native VRAM") as tracker:
        output_standard = fluid_layer(x, z)
    native_vram = tracker.results.get("cuda_peak", 0.0)
    
    # Restore original backend
    kernel_module.get_cuda_backend = orig_get_backend
    
    # 6. Try compiling/loading custom CUDA kernel
    cuda_backend = get_cuda_backend()
    
    cuda_avg_latency = 0.0
    cuda_vram = 0.0
    precision_loss = "N/A"
    cuda_available = False
    
    if cuda_backend is not None:
        cuda_available = True
        print("Benchmarking FluidLinear Custom CUDA Kernel...")
        # Warmup
        for _ in range(10):
            _ = fluid_layer(x, z)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(100): # Hardcoded to 100 as requested for CUDA
            _ = fluid_layer(x, z)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cuda_avg_latency = (time.perf_counter() - start_time) * 1000 / 100.0
        
        # VRAM check
        with MemoryTracker("FluidLinear CUDA VRAM") as tracker:
            output_fluid = fluid_layer(x, z)
        cuda_vram = tracker.results.get("cuda_peak", 0.0)
        
        # Numerical Correctness Validation
        precision_loss = torch.mean(torch.abs(output_standard - output_fluid)).item()
        
        # Assert closeness
        try:
            assert torch.allclose(output_standard, output_fluid, atol=1e-3), "Outputs are not close enough!"
            print("ASSERTION PASSED: outputs are mathematically close within 1e-3.")
        except AssertionError as e:
            print(f"ASSERTION FAILED: {e}")
            raise e
    else:
        print("\n[INFO] CUDA Kernel not loaded, skipping CUDA benchmark section.")
        output_fluid = None
        
    # 7. Print Table Output
    print("\n" + "=" * 90)
    print("PERFORMANCE BENCHMARK RESULTS TABLE (Hidden Dim: 8192)")
    print("=" * 90)
    print(f"{'Metric':<30} | {'Standard nn.Linear':<18} | {'Fluid PyTorch-Native':<20} | {'Fluid CUDA Custom':<18}")
    print("-" * 90)
    
    latency_native_str = f"{native_avg_latency:.2f} ms"
    latency_std_str = f"{std_avg_latency:.2f} ms"
    latency_cuda_str = f"{cuda_avg_latency:.2f} ms" if cuda_available else "N/A (No CUDA)"
    print(f"{'Average Latency (ms)':<30} | {latency_std_str:<18} | {latency_native_str:<20} | {latency_cuda_str:<18}")
    
    vram_std_str = f"{std_vram:.2f} MB" if device.type == "cuda" else "N/A (CPU)"
    vram_native_str = f"{native_vram:.2f} MB" if device.type == "cuda" else "N/A (CPU)"
    vram_cuda_str = f"{cuda_vram:.2f} MB" if (cuda_available and device.type == "cuda") else ("N/A (No CUDA)" if not cuda_available else "N/A (CPU)")
    print(f"{'Peak VRAM Usage (MB)':<30} | {vram_std_str:<18} | {vram_native_str:<20} | {vram_cuda_str:<18}")
    
    prec_std_str = "Baseline (Dense)"
    prec_native_str = "Baseline (Fluid)"
    prec_cuda_str = f"{precision_loss:.6e}" if cuda_available else "N/A"
    print(f"{'Precision Difference':<30} | {prec_std_str:<18} | {prec_native_str:<20} | {prec_cuda_str:<18}")
    print("=" * 90)

if __name__ == "__main__":
    run_cuda_validation_and_benchmark()
