#include <cuda.h>
#include <cuda_runtime.h>

// mainly to link rten.cu functions into main.cpp for debugging
namespace rten_cpp {
    at::Tensor sq5_gen_2d_cuda(int64_t M, int64_t N, int64_t seed);
    at::Tensor materialize_fwd_cuda(const at::Tensor& coef, int64_t seed, int64_t P, int64_t Q, int64_t stride_P, int64_t stride_Q);
}