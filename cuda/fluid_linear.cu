#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Fused CUDA forward kernel
__global__ void fluid_linear_forward_cuda_kernel(
    const float* __restrict__ X,              // [B * N, d_in]
    const float* __restrict__ V,              // [d_in, d_latent]
    const float* __restrict__ U,              // [d_out, d_latent]
    const float* __restrict__ bias,           // [d_out]
    const int* __restrict__ active_indices,   // [B, K]
    const float* __restrict__ active_vals,     // [B, K]
    float* __restrict__ Y,                     // [B * N, d_out]
    int B,
    int N,
    int d_in,
    int d_out,
    int d_latent,
    int K
) {
    // Dynamic Shared Memory Layout:
    // int s_indices[K]
    // float s_vals[K]
    // float s_H_mod[16 * K]
    extern __shared__ char s_mem[];
    int* s_indices = (int*)s_mem;
    float* s_vals = (float*)&s_indices[K];
    float* s_H_mod = (float*)&s_vals[K];

    int b = blockIdx.z; // batch element index
    int n_start = blockIdx.y * 16; // sequence dimension start index
    int j_start = blockIdx.x * 16; // output dimension start index
    
    int tx = threadIdx.x; // Thread X offset (0-15)
    int ty = threadIdx.y; // Thread Y offset (0-15)
    
    int thread_id = ty * blockDim.x + tx; // Thread index within block (0-255)
    int num_threads = blockDim.x * blockDim.y; // 256
    
    // 1. Load active metadata for the current batch element b into Shared Memory
    for (int k = thread_id; k < K; k += num_threads) {
        s_indices[k] = active_indices[b * K + k];
        s_vals[k] = active_vals[b * K + k];
    }
    __syncthreads();
    
    // 2. Compute modulated low-rank activations and store them in s_H_mod
    int total_elements = 16 * K;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int local_n = idx / K;
        int k = idx % K;
        int global_n = n_start + local_n;
        
        if (global_n < N) {
            int active_idx = s_indices[k];
            float val = 0.0f;
            for (int c = 0; c < d_in; ++c) {
                // X: (b * N + global_n) * d_in + c
                // V: c * d_latent + active_idx
                val += X[(b * N + global_n) * d_in + c] * V[c * d_latent + active_idx];
            }
            s_H_mod[local_n * K + k] = val * s_vals[k];
        } else {
            s_H_mod[local_n * K + k] = 0.0f;
        }
    }
    __syncthreads();
    
    // 3. Generate final output Y and add bias
    int global_n = n_start + ty;
    int global_j = j_start + tx;
    
    if (global_n < N && global_j < d_out) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int active_idx = s_indices[k];
            // U: global_j * d_latent + active_idx
            sum += s_H_mod[ty * K + k] * U[global_j * d_latent + active_idx];
        }
        // Write to global memory
        Y[(b * N + global_n) * d_out + global_j] = sum + bias[global_j];
    }
}

// C++ API Wrapper
torch::Tensor fluid_linear_forward_cuda(
    torch::Tensor x,               // [B, N, d_in]
    torch::Tensor V,               // [d_in, d_latent]
    torch::Tensor U,               // [d_out, d_latent]
    torch::Tensor bias,            // [d_out]
    torch::Tensor active_indices,  // [B, K]
    torch::Tensor active_vals      // [B, K]
) {
    // Ensure all tensors are contiguous and on the same device
    auto x_contig = x.contiguous();
    auto V_contig = V.contiguous();
    auto U_contig = U.contiguous();
    auto bias_contig = bias.contiguous();
    
    // Convert indices to int32 for optimized shared memory footprint
    auto active_indices_int = active_indices.to(torch::kInt32).contiguous();
    auto active_vals_contig = active_vals.contiguous();
    
    auto B = x_contig.size(0);
    auto N = x_contig.size(1);
    auto d_in = x_contig.size(2);
    auto d_out = U_contig.size(0);
    auto d_latent = U_contig.size(1);
    auto K = active_indices_int.size(1);
    
    // Reshape input tensor x to 2D
    auto x_flat = x_contig.view({B * N, d_in});
    
    // Allocate output tensor
    auto Y = torch::zeros({B * N, d_out}, x_contig.options());
    
    // Block size is fixed at 16x16 = 256 threads
    dim3 threads(16, 16);
    // Grid maps threads to cover N, d_out, and Batch (B)
    dim3 grid((d_out + 15) / 16, (N + 15) / 16, B);
    
    // Shared Memory size calculation
    size_t shared_mem_size = K * sizeof(int) + K * sizeof(float) + 16 * K * sizeof(float);
    
    // Launch CUDA Kernel
    fluid_linear_forward_cuda_kernel<<<grid, threads, shared_mem_size>>>(
        x_flat.data_ptr<float>(),
        V_contig.data_ptr<float>(),
        U_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        active_indices_int.data_ptr<int>(),
        active_vals_contig.data_ptr<float>(),
        Y.data_ptr<float>(),
        B,
        N,
        d_in,
        d_out,
        d_latent,
        K
    );
    
    // Reshape output back to 3D
    return Y.view({B, N, d_out});
}
