import os
import torch
from torch.utils.cpp_extension import load

_fluid_linear_cuda = None
_jit_attempted = False

def get_cuda_backend():
    """
    Attempts to compile and load the custom CUDA kernel for FluidLinear on-the-fly.
    Returns the module if successful, or None to trigger PyTorch-native fallback.
    """
    global _fluid_linear_cuda, _jit_attempted
    if _jit_attempted:
        return _fluid_linear_cuda
        
    _jit_attempted = True
    
    if not torch.cuda.is_available():
        print("[Neural Fluidity] CUDA device not available. Running in PyTorch-native mode.")
        return None
        
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    bindings_path = os.path.join(curr_dir, "cuda", "bindings.cpp")
    cuda_path = os.path.join(curr_dir, "cuda", "fluid_linear.cu")
    
    if not (os.path.exists(bindings_path) and os.path.exists(cuda_path)):
        print("[Neural Fluidity] CUDA source files not found. Running in PyTorch-native mode.")
        return None
        
    print("[Neural Fluidity] Compiling custom CUDA kernel JIT (Just-In-Time)...")
    try:
        # Load C++/CUDA extension dynamically
        _fluid_linear_cuda = load(
            name="fluid_linear_cuda",
            sources=[bindings_path, cuda_path],
            verbose=True,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3']
        )
        print("[Neural Fluidity] CUDA Kernel JIT compilation completed successfully.")
        return _fluid_linear_cuda
    except Exception as e:
        print(f"[Neural Fluidity] CUDA Kernel JIT compilation failed: {e}")
        print("[Neural Fluidity] Falling back to PyTorch-native implementation.")
        return None
