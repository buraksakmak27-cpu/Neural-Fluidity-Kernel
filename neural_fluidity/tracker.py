import torch
import os
import gc
import time

try:
    import psutil
except ImportError:
    psutil = None

class MemoryTracker:
    """
    Context manager to track peak memory usage (both VRAM and RAM) of a block of code.
    """
    def __init__(self, label="Block"):
        self.label = label
        self.start_cuda_peak = 0
        self.start_cuda_allocated = 0
        self.start_cpu = 0
        self.start_time = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_cpu_mem(self):
        if psutil is not None:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # MB
        return 0.0

    def __enter__(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_cuda_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        
        self.start_cpu = self._get_cpu_mem()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed = (end_time - self.start_time) * 1000  # ms
        
        gc.collect()
        
        cpu_peak = self._get_cpu_mem()
        cpu_diff = cpu_peak - self.start_cpu
        
        if self.device.type == "cuda":
            cuda_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            cuda_allocated_end = torch.cuda.memory_allocated() / (1024 * 1024)
            cuda_diff_peak = cuda_peak - self.start_cuda_allocated
            print(f"[{self.label}] CUDA Peak: {cuda_peak:.2f} MB | Net allocated change: {cuda_allocated_end - self.start_cuda_allocated:.2f} MB | Peak delta: {cuda_diff_peak:.2f} MB | CPU delta: {cpu_diff:.2f} MB | Time: {elapsed:.2f} ms")
            self.results = {
                "cuda_peak": cuda_peak,
                "cuda_allocated_change": cuda_allocated_end - self.start_cuda_allocated,
                "cuda_peak_delta": cuda_diff_peak,
                "cpu_delta": cpu_diff,
                "time_ms": elapsed
            }
        else:
            print(f"[{self.label}] (CPU Only) CPU delta: {cpu_diff:.2f} MB | Time: {elapsed:.2f} ms")
            self.results = {
                "cuda_peak": 0.0,
                "cuda_allocated_change": 0.0,
                "cuda_peak_delta": 0.0,
                "cpu_delta": cpu_diff,
                "time_ms": elapsed
            }
