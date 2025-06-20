#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "cuda/rten.cuh"

// mainly for debugging
int main() {
  // TODO: benchmark speed of constexpr vs #define
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor coef_tensor = torch::randn(32, options);
  std::cout << coef_tensor << std::endl;
  constexpr int seed = 1337;
  // 1<<10 = 2^10
  torch::Tensor out_tensor = rten_cpp::materialize_fwd_cuda(coef_tensor, seed,  1<<10,  1<<10, 1<<10, 1);
  // torch::Tensor out_tensor = rten_cpp::materialize_fwd_cuda(coef_tensor, seed, 256, 256);
  // std::cout << out_tensor << std::endl;

  torch::Tensor cpu_tensor = out_tensor.to(torch::kCPU);
  // assert foo is 2-dimensional and holds floats.
  auto foo_a = cpu_tensor.accessor<float,2>();
  for(int i = 0; i < 5; i++) {
    // use the accessor foo_a to get tensor data.
    std::cout << foo_a[i][i] << std::endl;
  }

  return 0;
}