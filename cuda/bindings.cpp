#include <torch/extension.h>

// Forward declaration of the CUDA implementation
torch::Tensor fluid_linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor V,
    torch::Tensor U,
    torch::Tensor bias,
    torch::Tensor active_indices,
    torch::Tensor active_vals
);

// Wrapper with device verification
torch::Tensor fluid_linear_forward(
    torch::Tensor x,
    torch::Tensor V,
    torch::Tensor U,
    torch::Tensor bias,
    torch::Tensor active_indices,
    torch::Tensor active_vals
) {
    if (x.device().is_cuda()) {
        return fluid_linear_forward_cuda(x, V, U, bias, active_indices, active_vals);
    } else {
        AT_ERROR("fluid_linear_forward is only implemented for CUDA device.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fluid_linear_forward, "FluidLinear forward pass (CUDA)");
}
