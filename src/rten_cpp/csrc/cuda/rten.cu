#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "rten.cuh"

namespace rten_cpp {
// this file follows (m, k) @ (k, n) = (m, n) for matmul to align with general convention

// implementation based on http://eiserloh.net/noise/SquirrelNoise5.hpp
// squirrel noise constants
constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
constexpr float ONE_OVER_MAX_INT = 1.0 / (float) 0x7FFFFFFF; // original code uses doubles for this rescale factor, but floats work just as well with sufficient precision

// #define SQ5_BIT_NOISE1 0xd2a80a3f;	// 11010010101010000000101000111111
// #define SQ5_BIT_NOISE2 0xa884f197;	// 10101000100001001111000110010111
// #define SQ5_BIT_NOISE3 0x6C736F4B;  // 01101100011100110110111101001011
// #define SQ5_BIT_NOISE4 0xB79F3ABB;	// 10110111100111110011101010111011
// #define SQ5_BIT_NOISE5 0x1b56c4f5;	// 00011011010101101100010011110101
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__device__ __forceinline__ float sq5_2d( int m_idx, int n_idx, unsigned int seed ) {
  /*  
  To mix the rows+cols, we elementwise multiply rows (m-th dim) by PRIME_NUMBER and elementwise broadcast and add onto cols (n-th dim).
  This mixes the rows+cols uniquely, forming a 2d [m (rows), n (cols)] shaped tensor, from which we generate the subset of the noise matrix.
  */
	unsigned int mangledBits = (unsigned int) (PRIME_NUMBER * m_idx + n_idx);
	mangledBits *= SQ5_BIT_NOISE1;
	mangledBits += seed;
	mangledBits ^= (mangledBits >> 9);
	mangledBits += SQ5_BIT_NOISE2;
	mangledBits ^= (mangledBits >> 11);
	mangledBits *= SQ5_BIT_NOISE3;
	mangledBits ^= (mangledBits >> 13);
	mangledBits += SQ5_BIT_NOISE4;
	mangledBits ^= (mangledBits >> 15);
	mangledBits *= SQ5_BIT_NOISE5;
	mangledBits ^= (mangledBits >> 17);

  // rescale the noise vector between -1 and 1
	return (float)( ONE_OVER_MAX_INT * (float) (int) mangledBits );
}

__global__ void sq5_gen_2d_kernel(float* out, int M, int N, unsigned int seed){
  // 2d generate M x N noise matrix
  // expects idx.x to be n-dim and idx.y to be m-dim of a (m, n) tensor
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int m_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (m_idx < M && n_idx < N) out[m_idx * M + n_idx] = sq5_2d(m_idx, n_idx, seed);
}

at::Tensor sq5_gen_2d_cuda(int64_t M, int64_t N, int64_t seed) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor out = torch::empty({M, N}, options);
  float* out_ptr = out.mutable_data_ptr<float>();

  const int blocksize_m = 8;
  const int blocksize_n = 8;

  // ceildiv + reversed (m,n) axis order because of xyz
  dim3 dimGrid(CEIL_DIV(N, blocksize_n), CEIL_DIV(M, blocksize_m));
  dim3 dimBlock(blocksize_n, blocksize_m);
  sq5_gen_2d_kernel<<<dimGrid, dimBlock>>>(out_ptr, M, N, seed);
  return out;
}


__global__ void materialize_fwd_kernel( float* out, const float* coef, unsigned int seed, int N, int M, int Q, int stride_P, int stride_Q ){
  // allocate buffer for entire coef vector in smem
  extern __shared__ float coef_s[];

  // each thread first loads a value of coef. force unroll since coefs should be small.
  #pragma unroll
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    // masking
    if (i < N) coef_s[i] = coef[i];
  }
  // block threads until coef_s is fully populated
  __syncthreads();

  // implements split-k type loop so that we don't unnecessarily materialize too many coef_s
  for (int blkIdx = blockIdx.x; blkIdx < CEIL_DIV(M, blockDim.x); blkIdx += gridDim.x){
    // m_idx is the idx of the current elem in the flattened (P, Q) out matrix that we "sequentially" iterate through
    int m_idx = blkIdx * blockDim.x + threadIdx.x;
    // masking ensures output is actually inside the out tensor
    if (m_idx < M){
      // while we efficiently iterate through contiguous indices of the out matrix, due to striding of the output 
      // matrix (i.e. transposed order), the actual m_idx passed into the sq5 func differs
      // `(m_idx % Q) * stride_Q` computes strided offset within the row
      // `(m_idx // Q) * stride_P` computes which row (P-dim) the element is on
      int sq_m_idx = (m_idx % Q) * stride_Q + (m_idx / Q) * stride_P;

      // iterate through coef and perform dot product
      float acc = 0.0f;
      for (int n = 0; n < N; n++){
        // this particular load of coef_s[n] should be done via warp broadcast since every other thread should be accessing the same n
        acc = fmaf(sq5_2d(sq_m_idx, n, seed), coef_s[n], acc);
      }
      // write out the dot product
      out[m_idx] = acc;
    }

  }
}

at::Tensor materialize_fwd_cuda(const at::Tensor& coef, int64_t seed, int64_t P, int64_t Q, int64_t stride_P, int64_t stride_Q){
  /*
  Fused kernel utilizing squirrel5 noise function to materialize a M -> (P, Q) sized output vector. 
  Will also work for 1d shaped (1, Q) dim vectors by setting stride_Q = 1.
  The out_ptr is the output vector of shape (P, Q), N = P * Q
  The coef_ptr is the coefficient vector of shape (N,)
  The noise function should lazily materialize the (M, N) matrix row by row (along M dim), each row getting fused dot product'd with
  the entire coef matrix, which should result in cache hits every time because the coef vector should be perma loaded into SRAM.
  The resulting equation is:    noise @ coef = out
  
  TODO: RAND_BIAS=1 to add fused random bias, turning eqn into: noise @ coef + scale_bias * bias = out

  coef vector should probably be smem. all threads cooperatively load coef vector into smem, then the threads each take a
  row along the M-dim and perform dot product between row and coef vector to obtain a single result output. This might end up
  causing frequent smem bank conflicts, however we can actually do warp broadcasts if all the threads access the same memory location in smem,
  so just make sure every thread accesses the same elems of the coef vector as they iterate through for the dot product.
  */
  // check coefs is one dimensional
  TORCH_CHECK(coef.dim() == 1);
  TORCH_CHECK(coef.is_contiguous());
  TORCH_INTERNAL_ASSERT(coef.device().type() == at::DeviceType::CUDA);
  const int M = P * Q;
  int N = coef.numel();
  const float* coef_ptr = coef.const_data_ptr<float>();

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor out = torch::empty({P, Q}, options);
  float* out_ptr = out.mutable_data_ptr<float>();

  // for now this is manually tuned. I've tried using cudaOccupancyMaxPotentialBlockSize but it doesn't give
  // good grid or block sizes at all. 
  const int BLOCKSIZE = 128;
  // launches NUM_SPLITS number of blocks which each load coef into smem
  const int NUM_SPLITS = 256;
  int smem = N*sizeof(float);

  dim3 dimGrid(NUM_SPLITS);
  dim3 dimBlock(BLOCKSIZE);
  materialize_fwd_kernel<<<dimGrid, dimBlock, smem>>>(out_ptr, coef_ptr, seed, N, M, Q, stride_P, stride_Q);
  return out;
}

// registers implementation for both CUDA and CPU
// there's technically no CPU implementation but I'm lazy to create one just to register it to throw an error
TORCH_LIBRARY_IMPL(rten_cpp, CompositeExplicitAutograd, m) {
  m.impl("sq5_gen_2d", &sq5_gen_2d_cuda);
  m.impl("materialize_fwd", &materialize_fwd_cuda);
}

}