"""
This file contains ver2 implementation of RandumbTensor that should more smoothly integrate with PyTorch autograd using an implementation schema 
similar to https://github.com/albanD/subclass_zoo/blob/main/negative_tensor.py (many thanks to albanD who pointed me in the right direction)

In essence, we wrap the creation of the RandumbTensor in a custom pytorch function which returns the RandumbTensor class given the input
torch.Tensor coef, seed and shape. Then we handle the backward pass of gradients back into the coef tensor manually via the custom function's bwd.
The custom RandumbTensor class here simply implements the __torch_dispatch__ custom behaviour (materialization and dematerialization) for the 
tensor for one operation, after which it returns a normal Tensor

With the original version rawdogging the tensor implementation without a torch Function wrapping the instantiation, there were many issues with the
tensor wrapping breaking the autograd computation graph. There was a lot of serious runtime overhead trying to manually stitch together the two 
computation graph bwd functions with grad hooks, which was extremely inelegant and generally buggy.

Note that the although not technically necessary materialize_fwd is registered as a pytorch function and works with autograd out of the box. 
This is mostly a legacy feature from v1 implementation, but given that torch compile and triton have many undocumented bugs with torch.Function 
composability I've kept it as is to prevent any unforseen bugs.

Squirrel5 noise based prng generation implementation is adapted from http://eiserloh.net/noise/SquirrelNoise5.hpp.
"""

from typing import Union
from collections.abc import Iterable
import os
import itertools
import math

from .utils import getBack, no_dispatch
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
# os.environ["TRITON_INTERPRET"] = "0"
# os.environ["TRITON_ALWAYS_COMPILE"] = "1"

# manual autotune flag. Will use the triton.autotune if set, otherwise manually pass in block sizes
if "AUTOTUNE" not in os.environ: os.environ["AUTOTUNE"] = "1"

import torch
from torch.utils._pytree import tree_map, tree_map_only
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton, custom_op

from torch.autograd import Function

import triton
import triton.language as tl

from torch.profiler import profile, record_function
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import detect_fake_mode


def construct_configs(cfgs: dict, min_blocks: int, max_blocks: int, *args, **kwargs):
	"""
	Helper function for constructing triton autotune configs.
	Example usage: 
 	configs = construct_configs({"BLOCK_SIZE_N": [2**i for i in range(4, 7)], 
								"BLOCK_SIZE_M": [2**i for i in range(4, 7)], 
								"SPLIT_N": [2**i for i in range(4, 7)]}, None, 32 * 32 * 16 + 1, num_warps=4)
	Args:
		cfgs (dict): key should be block_size, and the value a list of possible values
		min_blocks (int): only include configs where total product of block_sizes is more than or equal this
		max_blocks (int): only include configs where total product of block_sizes is less than or equal this

	Returns:
		list of possible triton.Config objects
	"""
	if min_blocks is None: min_blocks = -math.inf
	if max_blocks is None: max_blocks = math.inf
	permutations = itertools.product(*cfgs.values())
	configs = []
	for values in permutations:
		prod = math.prod(values)
		if prod >= min_blocks and prod <= max_blocks:
			configs.append(triton.Config(kwargs=dict(zip(cfgs.keys(), values)), *args, **kwargs))
	return configs


# the following are standalone helper kernel implementations, used mainly for crosschecking correctness of the more complex implementations
@triton.jit
def squirrel5_kernel_2d(out_ptr,    # ptr to output vector (N, M)
                N, M,   # sizes of the vectors
                seed,   # seed for the generated tensor
                stride_n, stride_m,
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    Helper kernel materializing only the noise matrix (for debugging)
    The out_ptr is the noise matrix of shape (N, M)
    """
    # identifies correct memory address of the part of the output vector we are writing to
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = (offs_n < N)[:, None] & (offs_m < M)[None, :]
    
    # squirrel5 noise function
    mangledBits = squirrel5_gen_2d(offs_m, offs_n, seed)   # [BLOCK_SIZE_N, BLOCK_SIZE_M]
    out_ptrs = out_ptr + stride_n * offs_n[:, None] + stride_m * offs_m[None, :]
    
    tl.store(out_ptrs, mangledBits, mask=mask)

def squirrel5_generate_2d(N, M, seed=1337):
    output = torch.empty(size=(N, M), device='cuda')
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    squirrel5_kernel_2d[grid](output, N, M, seed, output.stride(0), output.stride(1))
    return output


@triton.jit
def squirrel5_kernel_1d(out_ptr, N, seed, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N
    mangledBits = squirrel5_gen_1d(offs, seed)
    tl.store(out_ptr + offs, mangledBits, mask=mask)
    
if int(os.environ["AUTOTUNE"]): 
    squirrel5_kernel_2d = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [2**i for i in range(2, 8)],
                                                                     "BLOCK_SIZE_M": [2**i for i in range(2, 8)]}, 4 * 64, None, num_warps=4),
                                                                 key=['N', 'M'])(squirrel5_kernel_2d)
    squirrel5_kernel_1d = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [2**i for i in range(2, 8)]}, None, None, num_warps=4), key=['N'])(squirrel5_kernel_1d)

def squirrel5_generate_1d(N, seed=1337):
    output = torch.empty(size=(N,), device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    squirrel5_kernel_1d[grid](output, N, seed)
    return output


# trition kernel implementations for the tensor
@triton.jit
def squirrel5_gen_2d(cols: tl.tensor, rows: tl.tensor, seed):
    """    
    rows is an array of indices expected from leading n-dim. Indices of the rows of the noise matrix we are generating.
    cols is an array of indices expected from trailing m-dim. Indices of the columns of the noise matrix we are generating.
    
    To mix the rows+cols, we elementwise multiply rows (n-th dim) by PRIME_NUMBER and elementwise broadcast and add onto cols (m-th dim).
    This mixes the rows+cols uniquely, forming a 2d [n (rows), m (cols)] shaped tensor, from which we generate the subset of the noise matrix.
    """
    # squirrel noise constants
    SQ5_BIT_NOISE1: tl.constexpr = 0xd2a80a3f	# 11010010101010000000101000111111
    SQ5_BIT_NOISE2: tl.constexpr = 0xa884f197	# 10101000100001001111000110010111
    SQ5_BIT_NOISE3: tl.constexpr = 0x6C736F4B # 01101100011100110110111101001011
    SQ5_BIT_NOISE4: tl.constexpr = 0xB79F3ABB	# 10110111100111110011101010111011
    SQ5_BIT_NOISE5: tl.constexpr = 0x1b56c4f5	# 00011011010101101100010011110101
    PRIME_NUMBER: tl.constexpr = 198491317 # Large prime number with non-boring bits
    
    #TODO: this line causes long scoreboard warp stalls, but there's no L1 data access??
    mangledBits = (PRIME_NUMBER * rows.to(tl.uint32))[:, None] + cols.to(tl.uint32)[None, :]
    mangledBits *= SQ5_BIT_NOISE1
    mangledBits += seed
    mangledBits ^= (mangledBits >> 9)
    mangledBits += SQ5_BIT_NOISE2
    mangledBits ^= (mangledBits >> 11)
    mangledBits *= SQ5_BIT_NOISE3
    mangledBits ^= (mangledBits >> 13)
    mangledBits += SQ5_BIT_NOISE4
    mangledBits ^= (mangledBits >> 15)
    mangledBits *= SQ5_BIT_NOISE5 
    mangledBits ^= (mangledBits >> 17)
    
    # rescale the noise vector between -1 and 1
    ONE_OVER_MAX_INT: tl.constexpr = 1.0 / 0x7FFFFFFF
    mangledBits = ( ONE_OVER_MAX_INT * mangledBits.to(tl.int32).to(tl.float32) )
    return mangledBits

@triton.jit
def squirrel5_gen_2d_T(cols: tl.tensor, rows: tl.tensor, seed):
    """    
    Transposed variation of the squirrel5_gen_2d algorithm, utilised in the backward pass. Note the transposition of the n, m matrices in the rows & cols definition below.
    
    rows is an array of indices expected from leading m-dim. Indices of the rows of the transposed noise matrix we are generating.
    cols is an array of indices expected from trailing n-dim. Indices of the columns of the transposed noise matrix we are generating.
    
    To mix the rows+cols, we elementwise multiply rows (m-th dim) by PRIME_NUMBER and elementwise broadcast and add onto cols (n-th dim).
    This mixes the rows+cols uniquely, forming a 2d [m (rows), n (cols)] shaped tensor, from which we generate the subset of the transposed noise matrix.
    """
    # squirrel noise constants
    SQ5_BIT_NOISE1: tl.constexpr = 0xd2a80a3f	# 11010010101010000000101000111111
    SQ5_BIT_NOISE2: tl.constexpr = 0xa884f197	# 10101000100001001111000110010111
    SQ5_BIT_NOISE3: tl.constexpr = 0x6C736F4B # 01101100011100110110111101001011
    SQ5_BIT_NOISE4: tl.constexpr = 0xB79F3ABB	# 10110111100111110011101010111011
    SQ5_BIT_NOISE5: tl.constexpr = 0x1b56c4f5	# 00011011010101101100010011110101
    PRIME_NUMBER: tl.constexpr = 198491317 # Large prime number with non-boring bits
    
    mangledBits = (PRIME_NUMBER * rows.to(tl.uint32))[None, :] + cols.to(tl.uint32)[:, None]
    mangledBits *= SQ5_BIT_NOISE1
    mangledBits += seed
    mangledBits ^= (mangledBits >> 9)
    mangledBits += SQ5_BIT_NOISE2
    mangledBits ^= (mangledBits >> 11)
    mangledBits *= SQ5_BIT_NOISE3
    mangledBits ^= (mangledBits >> 13)
    mangledBits += SQ5_BIT_NOISE4
    mangledBits ^= (mangledBits >> 15)
    mangledBits *= SQ5_BIT_NOISE5 
    mangledBits ^= (mangledBits >> 17)
    
    # rescale the noise vector between -1 and 1
    ONE_OVER_MAX_INT: tl.constexpr = 1.0 / 0x7FFFFFFF
    mangledBits = ( ONE_OVER_MAX_INT * mangledBits.to(tl.int32).to(tl.float32) ).to(tl.float32)
    return mangledBits


@triton.jit
def squirrel5_gen_1d(idx: tl.tensor, seed):
    """    
    1d version. only idx, self-explanatory. used for the fused random bias.
    """
    # squirrel noise constants
    SQ5_BIT_NOISE1: tl.constexpr = 0xd2a80a3f	# 11010010101010000000101000111111
    SQ5_BIT_NOISE2: tl.constexpr = 0xa884f197	# 10101000100001001111000110010111
    SQ5_BIT_NOISE3: tl.constexpr = 0x6C736F4B # 01101100011100110110111101001011
    SQ5_BIT_NOISE4: tl.constexpr = 0xB79F3ABB	# 10110111100111110011101010111011
    SQ5_BIT_NOISE5: tl.constexpr = 0x1b56c4f5	# 00011011010101101100010011110101
    
    mangledBits = idx.to(tl.uint32)
    mangledBits *= SQ5_BIT_NOISE1
    mangledBits += seed
    mangledBits ^= (mangledBits >> 9)
    mangledBits += SQ5_BIT_NOISE2
    mangledBits ^= (mangledBits >> 11)
    mangledBits *= SQ5_BIT_NOISE3
    mangledBits ^= (mangledBits >> 13)
    mangledBits += SQ5_BIT_NOISE4
    mangledBits ^= (mangledBits >> 15)
    mangledBits *= SQ5_BIT_NOISE5 
    mangledBits ^= (mangledBits >> 17)
    
    # rescale the noise vector between -1 and 1
    ONE_OVER_MAX_INT: tl.constexpr = 1.0 / 0x7FFFFFFF
    mangledBits = ( ONE_OVER_MAX_INT * mangledBits.to(tl.int32).to(tl.float32) )
    return mangledBits


@triton.jit
def materialize_fwd_kernel(out_ptr,    # ptr to output vector
                coef_ptr,  # ptr to the coefficients vector
                N, M,   # sizes of the vectors
                seed,   # seed for the generated tensor
                P, Q,	# final shape of output vector
                stride_P, stride_Q, # strides of output vector
                scale_bias_ptr, RAND_BIAS: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    Fused kernel utilizing squirrel5 noise function to materialize a N -> (P, Q) sized output vector. 
    Will also work for 1d shaped (1, Q) dim vectors by setting stride_Q = 1.
    The out_ptr is the output vector of shape (P, Q), N = P * Q
    The coef_ptr is the coefficient vector of shape (M,)
    The noise function should lazily materialize the (N, M) matrix row by row (along N dim), each row getting fused dot product'd with
    the entire coef matrix, which should result in cache hits every time because the coef matrix should be perma loaded into SRAM.
    The resulting equation is:    noise @ coef = out
    
    RAND_BIAS=1 to add fused random bias, turning eqn into: noise @ coef + scale_bias * bias = out
    """
    # identifies correct memory address of the part of the output vector we are writing to
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N
    
    # it is more efficient to write/tl.store contiguous indices of the matrix, but this means that the actual offs_n 
    # will need more complicated computation if stride_Q != 1 (i.e. transposed matrix), instead we have to use:
    # `(offs_n % Q) * stride_Q` computes strided offset within the row
    # `(offs_n // Q) * stride_P` computes which row (P-dim) the element is on
    sq_offs_n = (offs_n % Q) * stride_Q + (offs_n // Q) * stride_P
    
    # iterate along M-dim. These aren't "true" blocks in the sense that computation for blocks is split between pids, all chunks of computation here
    # are done in the same pid/block, this is more like "phases" within a single block, beacuse we require value of the entire vector to compute the dot product.
    accumulator = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        
        # squirrel5 noise function
        mangledBits = squirrel5_gen_2d(offs_m, sq_offs_n, seed)   # [BLOCK_SIZE_N, BLOCK_SIZE_M]
        
        # mask off mangledBits that are > M, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_M = offs_m < M
        # load the subgroup of coefficients we are processing
        coefs = tl.load(coef_ptr + offs_m, mask=mask_M)[None, :]  # [1, BLOCK_SIZE_M]
                
        # dot product via broadcast of elementwise multiplication of coefs across N-dim of materialized noise matrix
        # followed by summation along M-dim
        accumulator += tl.sum(mangledBits * coefs, axis=1)  # [BLOCK_SIZE_N,]
        
    # add fused random bias
    if RAND_BIAS == 1:
        scale_bias = tl.load(scale_bias_ptr)
        accumulator += scale_bias * squirrel5_gen_1d(sq_offs_n, seed)
    tl.store(out_ptr + offs_n, accumulator, mask=mask)

# TODO: while it's not an immediate concern, the grads that coef tensors have is extremely large for most normal model N values,
# since the grads are additive and any small sized downproj dim will inevitably accumulate very big grad values (since grads summed along N-dim)
# this might cause unstable training dynamics and could be the reason for the initial spike in loss that happens in most training runs.

# torch.empty() initialization needs to be zerod out since we use tl.atomic_add instead of tl.store
def _zero_output_coef(*args, **kwargs):
    if isinstance(args[0], dict): 
        args[0]["dcoef_ptr"].zero_()
    else:
        args[0].zero_()
	
 
@triton.jit
def materialize_bwd_kernel(dcoef_ptr,    # ptr to dcoef vector
                dout_ptr,  # ptr to the dout vector
                N, M,   # sizes of the vectors
                seed,   # seed for the generated tensor
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, SPLIT_N: tl.constexpr):
    """    
    Backwards pass of the above materialization kernel. This one will compute dcoef via dout and generating the transposed noise matrix.
    Since the noise matrix is transposed, the dims of it are (m, n) instead.
    given the eqn: noise @ coef = out |   (n, m) @ (m, 1) = (n, 1)
    dcoef = noise.T @ dout    |   (m, 1) = (m, n) @ (n, 1)
    We generate the transpose of the noise matrix and fuse the matmul. Unfortunately because usually m << n the naive algorithm becomes very
    inefficient. Implements split-k along N dim, in order to get coalesced loads we want to interleave the blocks such that each pid 
    simultaneously loads consecutive blocks during each for loop iteration. The N dimension is split into (BLOCK_SIZE_N * SPLIT_N) sized 
    coalesced loads, for a total number of N // (BLOCK_SIZE_N * SPLIT_N) loads. Each pid should process a separate BLOCK_SIZE_N block of 
    a particular load.
    Unlike in the fwd pass, we don't actually have to care about the striding, because the dot product is performed along the n-dim on all 
    elements of the RT, which means the specific order of elements doesn't matter since the summation of all dot products along n-dim is 
    permutation-invariant.
    """
    # identifies correct memory address of the part of the dcoef vector we are writing to
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    # start at first block to load
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)    
    # these are just compiler hints, note multiple_of documentation is wrong, ref: https://github.com/triton-lang/triton/issues/1324
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # size of each coalesced load
    LOAD_SIZE = BLOCK_SIZE_N*SPLIT_N
    # iterate along N-dim (of the "transposed" noise matrix)
    accumulator = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    # n-th coalesced load of size BLOCK_SIZE_N * SPLIT_N
    for n in range(0, tl.cdiv(N, LOAD_SIZE)):
        # squirrel5 noise function
        mangledBits = squirrel5_gen_2d_T(offs_m, offs_n, seed)   # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        
        # mask off mangledBits that are > N, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_N = offs_n < N
        # load the subgroup of dout we are processing
        dout = tl.load(dout_ptr + offs_n, mask=mask_N)[None, :]  # [1, BLOCK_SIZE_N]
        
        # dot product via broadcast of elementwise multiplication of dout across M-dim of materialized noise matrix
        # followed by summation along N-dim
        accumulator += tl.sum(mangledBits * dout, axis=1) # [BLOCK_SIZE_M,]
        
        # jump to next block to load
        offs_n += LOAD_SIZE
    
    accumulator = accumulator.to(dcoef_ptr.dtype.element_ty)
    tl.atomic_add(dcoef_ptr + offs_m, accumulator, mask=mask_m, sem="relaxed")

if int(os.environ["AUTOTUNE"]): 
    materialize_fwd_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [32, 64, 256, 1024],
                                                                        "BLOCK_SIZE_M": [8, 16, 32]}, None, None, num_warps=4),
                                             key=['N', 'M'])(materialize_fwd_kernel)
    materialize_bwd_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [2**i for i in range(8, 11)],
                                                                        "BLOCK_SIZE_M": [4, 16], "SPLIT_N": [4, 8, 16]}, None, None, num_warps=4, pre_hook=_zero_output_coef),
                                             key=['N', 'M'])(materialize_bwd_kernel)
else:
    materialize_bwd_kernel.add_pre_run_hook(_zero_output_coef)

@triton_op("rten::materialize_fwd", mutates_args={})
def materialize_fwd(coefs: torch.Tensor, size: list[int], strides: list[int], seed: int, scale_bias: torch.Tensor, fused_bias: bool) -> torch.Tensor:
    assert not (fused_bias and scale_bias is None), "scale_bias not initialized but fused_bias is True"
    output = torch.empty(size=size, device=coefs.device)
    if len(size) == 1:
        size = (1, size[0])
        strides = (0, strides[0])
    N = size[0] * size[1]
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    materialize_fwd_kernel[grid](output, coefs, N, coefs.numel(), seed, *size, *strides, scale_bias, int(fused_bias))
    return output

@triton_op("rten::materialize_bwd", mutates_args={})
def materialize_bwd(dout: torch.Tensor, dcoefs_size: int, seed: int) -> torch.Tensor:
    # note to self: should we enforce dout to be a flattened tensor? it doesn't seem to be necessary
    assert dout.is_contiguous(), "dout is not contiguous, backwards pass might give incorrect results."
    dcoefs = torch.empty(size=(dcoefs_size,), device=dout.device)
    grid = lambda meta: (triton.cdiv(dcoefs_size, meta['BLOCK_SIZE_M']), meta['SPLIT_N'])
    wrap_triton(materialize_bwd_kernel)[grid](dcoefs, dout, dout.numel(), dcoefs_size, seed)
    return dcoefs

# seperate backward pass to compute the grad for the scale_bias if it is used
def _zero_output_scale_bias(*args, **kwargs):
    if isinstance(args[0], dict): 
        args[0]["dbias_ptr"].zero_()
    else:
        args[0].zero_()
 
@triton.jit
def scale_bias_bwd_kernel(dbias_ptr, dout_ptr, N, seed, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N
    dout = tl.load(dout_ptr + offs, mask=mask)
    accumulator = tl.sum(dout * squirrel5_gen_1d(offs, seed))
    accumulator = accumulator.to(dbias_ptr.dtype.element_ty)
    tl.atomic_add(dbias_ptr, accumulator, sem="relaxed")

if int(os.environ["AUTOTUNE"]):
    scale_bias_bwd_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [2**i for i in range(8, 11)]}, None, None, num_warps=4, pre_hook=_zero_output_scale_bias), key=["N"])
else:
    scale_bias_bwd_kernel.add_pre_run_hook(_zero_output_scale_bias)

@triton_op("rten::scale_bias_bwd", mutates_args={})
def scale_bias_bwd(dout: torch.Tensor, seed: int) -> torch.Tensor:
    assert dout.is_contiguous(), "dout is not contiguous, backwards pass might give incorrect results."
    dbias = torch.empty(size=(1,), device=dout.device)
    grid = lambda meta: (triton.cdiv(dout.numel(), meta['BLOCK_SIZE_N']),)
    wrap_triton(scale_bias_bwd_kernel)[grid](dbias, dout, dout.numel(), seed)
    return dbias

# note: the backward autograd registration here isn't intended to be called, all RTs are constructed via RandumbTensorConstructor which have their own backward function defined
def _mat_backward(ctx, grad):
    dcoefs = materialize_bwd(grad, ctx.dcoefs_size, ctx.seed)
    dscale_bias = None
    if ctx.fused_bias: dscale_bias = scale_bias_bwd(grad, ctx.seed)
    # only coefs needs grad, the rest are non-differentiable. function signature of fwd: coefs, size, seed, scale_bias, fused_bias
    return dcoefs, None, None, dscale_bias, None

def _mat_setup_context(ctx, inputs, output):
    coefs, size, seed, scale_bias, fused_bias = inputs
    ctx.dcoefs_size = coefs.numel()
    ctx.seed = seed
    ctx.fused_bias = fused_bias

materialize_fwd.register_autograd(_mat_backward, setup_context=_mat_setup_context)

@materialize_fwd.register_kernel("cpu")
def _mat_fwd(coefs: torch.Tensor, size: int, seed: int, fused_bias: bool):
    raise NotImplementedError("Unsupported CPU implementation of materialize_fwd called.")

@materialize_bwd.register_kernel("cpu")
def _mat_bwd(dout: torch.Tensor, dcoefs_size: int, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of materialize_bwd called.")

@scale_bias_bwd.register_kernel("cpu")
def _scale_bias_bwd(dout: torch.Tensor, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of scale_bias_bwd called.")


def _zero_output_outptr(*args, **kwargs):
    if isinstance(args[0], dict): 
        args[0]["out_ptr"].zero_()
    else:
        args[0].zero_()

@triton.jit
def rt_mm_kernel(out_ptr,    # ptr to output vector
                coef_ptr,  # ptr to the coefficients vector
                m1_ptr, # ptr to the other other tensor mat1
                M,   # size of the RT int dim
                P, Q,	# size of the RT (P, Q)
                L,	# size of the other tensor mat1 (Q, L)
                stride_rt_P, stride_rt_Q, # strides of RT
                stride_m1_L, stride_m1_P,	# strides of other tensor mat1
                stride_out_L, stride_out_Q,
                seed,   # seed for the generated tensor
                #TODO: scale bias fused
                scale_bias_ptr, RAND_BIAS: tl.constexpr,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_L: tl.constexpr, SPLIT_L: tl.constexpr):
    """
    Fused kernel computing the matmul eqn: mat1 @ RT = out | (L, P) @ (P, Q) = (L, Q)
    
    The cost of materializing a column of the RT is the bitshift squirrel5 kernel and a dot product against a coef vector which should already be in L1. It should be confirmed and benched the perf of L1 materialization vs loading
    an already materialized variant of the column from L2. 
    Here we allocate certain pids to specific columns of the RT and maintain the pid persistently to keep the column in L1 cache, then load the required row from mat1. But this results in very bad mem access pattern?
    Maybe each pid assigned to 1 col of RT, materialize it in L1, then load first row of mat1, then second row, sequentially, until run out of rows, then move onto the next block of columns and repeat per each row?
    This would write consecutive indices of out matrix too, first row of out mat, then second row etc. for coalesced writes. This means that every pid at the same iterations would mem read the same row of mat1 and the
    mem read would be broadcasted to all the L1 caches making it faster.
    
    In torch the usual memory format for tensors is row-major, which is slightly more inefficient when accessing elements for mat2 (hence some ops
    pass in mat2.t() like linear layer). This is completely fine if mat2 is an RT, since in that case we can compute the output in row-major without
    loading the whole mat2 in memory, instead of having to group-order the output, increasing chances for coalesced writes.
    
    For each elem in output C, we load the corresponding mat1 row (1,P) into SRAM, then compute the respective noise matrix for a column of B (P, M).
    The noise matrix materialized via squirrel5_gen_2d requires offs_n, which is the indices of P modified by the strides.
    Then run a dot product along P-dim against the coef vector (M,) to get a vector of the column B (P,) which we then dot product with mat1 row (1,P).
    The two dot products can actually be fused, but might be more efficient to let compiler handle that.
    
    This approach even though optimized is still really bad, not faster, and doesn't scale better. DO NOT USE THIS, INSTEAD USE THE CUDA KERNEL (TO BE COMPLETED IN THE FUTURE).
    """
    # identifies consecutive part of the output vector we are writing to (can be across L-dim)
    # different pids of L will regenerate the col, so there will be SPLIT_L number of times that the RT col is recomputed,
    # which can further be split into BLOCK_SIZE_L for loop within a pid. But this is for future work, the idea is that rows in mat1
    # remain in L2 cache and are not completely flushed. This should follow the grouped ordering in triton docs? Won't work because
    # multiple warps/blocks will attempt to read the same row of mat1 at the same time, and broadcasts cannot be serialised to multiple
    # warps. This is probably a big bottleneck. However we want to minimize the number of "reaccesses" to the RT
    pid_q = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_l = tl.program_id(2)
    # offs_l = pid_l * GROUP_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    # mask_l = offs_l < L
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    mask_p = offs_p < P
    offs_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_q = tl.max_contiguous(tl.multiple_of(offs_q, BLOCK_SIZE_Q), BLOCK_SIZE_Q)
    mask_q = offs_q < Q

    col = tl.zeros([BLOCK_SIZE_P * BLOCK_SIZE_Q], dtype=tl.float32)
    offs_n = tl.ravel(offs_p[:, None] * stride_rt_P + offs_q[None, :] * stride_rt_Q)	# (BLOCK_SIZE_P, BLOCK_SIZE_Q)
    # squirrel5 noise function to generate a block of the column, that block being some consecutive BLOCK_SIZE_P values
    # loop has to be unrolled otherwise for some reason it makes 1byte requests to L1 cache?
    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_M), loop_unroll_factor=2, flatten=True):
        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mangledBits = squirrel5_gen_2d(offs_m, offs_n, seed)    # (BLOCK_SIZE_P * BLOCK_SIZE_Q, BLOCK_SIZE_M)
        coefs = tl.load(coef_ptr + offs_m, mask=offs_m < M, other=0.0)[None, :]    # (1, BLOCK_SIZE_M)
        col += tl.sum(mangledBits * coefs, axis=1)  # (BLOCK_SIZE_P * BLOCK_SIZE_Q)
    # offs_m = tl.arange(0, BLOCK_SIZE_M)
    # mangledBits = squirrel5_gen_2d(offs_m, offs_n, seed)    # (BLOCK_SIZE_P * BLOCK_SIZE_Q, BLOCK_SIZE_M)
    # # TODO: L2 local load pattern??? but coefs vector should be entirely small enough to fit into smem. maybe i guarantee it explicitly without this BLOCK_SIZE_M loop?
    # coefs = tl.load(coef_ptr + offs_m, mask=offs_m < M, other=0.0)[None, :]    # (1, BLOCK_SIZE_M)
    # col = tl.sum(mangledBits * coefs, axis=1)  # (BLOCK_SIZE_P * BLOCK_SIZE_Q)
        
    #TODO: i think there has to be masking for the col P,Q dims, but can't figure it out after reshape? might need tl.where
    col = tl.where(mask_p[:, None] & mask_q[None, :], tl.reshape(col, (BLOCK_SIZE_P, BLOCK_SIZE_Q), can_reorder=False), 0.0)
    # col = tl.reshape(col, (BLOCK_SIZE_P, BLOCK_SIZE_Q), can_reorder=False)
    # col = tl.zeros([BLOCK_SIZE_P, BLOCK_SIZE_Q], dtype=tl.float32)
        
    #chunked
    CHUNK_SIZE = L // SPLIT_L
    offs_l = pid_l * CHUNK_SIZE + tl.arange(0, BLOCK_SIZE_L)
    row_ptrs = m1_ptr + offs_l[:, None] * stride_m1_L + offs_p[None, :] * stride_m1_P
    out_ptrs = out_ptr + offs_l[:, None] * stride_out_L + offs_q[None, :] * stride_out_Q
    
    for l in tl.range(0, tl.cdiv(CHUNK_SIZE, BLOCK_SIZE_L)):
        mask_l = offs_l < CHUNK_SIZE*(SPLIT_L+1) - l * BLOCK_SIZE_L
        row = tl.load(row_ptrs, mask=mask_l[:, None] & mask_p[None, :], other=0.0)
        # row = tl.zeros([BLOCK_SIZE_L, BLOCK_SIZE_P], dtype=tl.float32)
        dot_prod = tl.dot(row, col, allow_tf32=False)
        # TODO: atomic add causing uncoalesced global accesses. blocksize32 causing 8x32bit mem accesses
        # it seems relaxed is the lowest number of mem accesses at only 8. any other sem causes more.
        # but it is not documented that atomics are unable to coalesce because of acq_rel semantics?
        # by right a relaxed sem atomic add should allow coalesced writes.
        # consider inline_asm_elementwise to brute force the coalesced atomic write?
        # 4 element at a time fill up 128 byte with atomic red write ref: https://gist.github.com/bertmaher/e33b874f75cb82451060b88ee20b8203#file-fa-py-L301
        # tl.atomic_add(out_ptrs, dot_prod, mask=mask_l[:, None] & mask_q[None, :], sem="relaxed")
        # XD? if i tl.store the mem accesses are all coalesced into a single 128 byte transaction
        # tl.store(out_ptrs, dot_prod, mask=mask_l[:, None] & mask_q[None, :])
        
        # i can only guess ncu is complaining because im too poorge to afford gpu with compute capability >sm_90 that supports vectorised atomics
        # per https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
        # so triton just compiles the atomics into uncoalesced packed instruction set... instead of interleaving the atomic adds for every thread inside a warp?
        # my options are to:
        # 1. accept that triton is inefficient and suck it up
        # 2. write custom asm inline that implements interleaved atomic adds
        
        # mov.b32 $0, %tid.x;
        # this accesses tid.x index and assigns it to =r output register $0
        # probably also need %ntid.x?
        # compute some form of offset and interleaving with that
        _ = tl.inline_asm_elementwise(
            """
            red.global.relaxed.add.f32 [$4], $8;
            red.global.relaxed.add.f32 [$5], $9;
            red.global.relaxed.add.f32 [$6], $10;
            red.global.relaxed.add.f32 [$7], $11;
            """,
            (
                # 4 dummy output registers
                "=r,=r,=r,=r,"
                # outptr registers
                "l,l,l,l,"
                # dot product registers
                "r,r,r,r"
            ),
            args=[
                out_ptrs.to(tl.uint64),
                dot_prod.to(tl.float32),
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=4,
        )
        
        row_ptrs += BLOCK_SIZE_L * stride_m1_L
        out_ptrs += BLOCK_SIZE_L * stride_out_L
    
    ########################################################
    #interleaved 
    """
    # multiply the BLOCK_SIZE_P col values with corresponding BLOCK_SIZE_P row values along the same P offset
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    # offs_l = tl.max_contiguous(tl.multiple_of(offs_l, BLOCK_SIZE_L), BLOCK_SIZE_L)
    row_ptrs = m1_ptr + offs_l[:, None] * stride_m1_L + offs_p[None, :] * stride_m1_P
    out_ptrs = out_ptr + offs_l[:, None] * stride_out_L + offs_q[None, :] * stride_out_Q
    # out_ptrs = out_ptr + offs_l[:, None] * Q + offs_q[None, :]
    # it seems that global memory loads are not broadcasted to different warps if they all request the same row
    #TODO: the loading scheme for rows seems to be uncoalesced? 1 byte per sector used 
    # row = tl.load(row_ptrs, mask=mask_l[:, None] & mask_p[None, :], other=0.0)	# (BLOCK_SIZE_L, BLOCK_SIZE_P)
    
    # mask_out = mask_l[:, None] & mask_q[None, :]
    # dot_prod = tl.dot(row, col, allow_tf32=False)
    # # if allow_tf32 is True the fp32xfp32 dot product will be pretty significantly numerically different from correct values: https://github.com/triton-lang/triton/issues/4574
    # tl.atomic_add(out_ptrs, dot_prod, mask=mask_out, sem="relaxed")
    # interleaved. but idk why this helps at all, the interleave is across row 
    # acc = tl.zeros(BLOCK_SIZE_L * SPLIT_L, BLOCK_SIZE_Q)
    for l in range(0, tl.cdiv(L, BLOCK_SIZE_L*SPLIT_L)):
        # mask_l = offs_l < GROUP_SIZE_L - l * BLOCK_SIZE_L
        mask_l = offs_l < L - l * BLOCK_SIZE_L*SPLIT_L
        # it seems that global memory loads are not broadcasted to different warps if they all request the same row
        #TODO: the loading scheme for rows seems to be uncoalesced? 1 byte per sector used 
        row = tl.load(row_ptrs, mask=mask_l[:, None] & mask_p[None, :], other=0.0)	# (BLOCK_SIZE_L, BLOCK_SIZE_P)
        
        mask_out = mask_l[:, None] & mask_q[None, :]
        dot_prod = tl.dot(row, col, allow_tf32=False)
        # if allow_tf32 is True the fp32xfp32 dot product will be pretty significantly numerically different from correct values: https://github.com/triton-lang/triton/issues/4574
        tl.atomic_add(out_ptrs, dot_prod, mask=mask_out, sem="relaxed")
        # advance to the next row
        row_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_m1_L
        # out_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_out_L
        out_ptrs += BLOCK_SIZE_L * SPLIT_L * Q
    """
    
if int(os.environ["AUTOTUNE"]):
    rt_mm_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_M": [32],
                                                              "BLOCK_SIZE_P": [32],
                                                              "BLOCK_SIZE_Q": [32],
                                                              "BLOCK_SIZE_L": [32],
                                                              "SPLIT_L": [4]}, None, None, num_warps=4, pre_hook=_zero_output_outptr),
                                   key=['M', 'P', 'Q', 'L'])(rt_mm_kernel)
else:
    rt_mm_kernel.add_pre_run_hook(_zero_output_outptr)

def rt_mm(mat1: torch.Tensor, coefs: torch.Tensor, size: list[int], strides: list[int], seed: int, scale_bias: torch.Tensor, fused_bias: bool) -> torch.Tensor:
    M = coefs.size(0)
    P, Q = size
    L = mat1.size(0)
    assert mat1.size(1) == P, f"mm shape mismatch, received mat1 shape {mat1.size()} and rt shape {size}"
    assert not (fused_bias and scale_bias is None), "scale_bias not initialized but fused_bias is True"
    output = torch.empty(size=(L, Q), device=coefs.device)
    grid = lambda meta: ( triton.cdiv(Q, meta['BLOCK_SIZE_Q']), triton.cdiv(P, meta['BLOCK_SIZE_P']), meta['SPLIT_L'] )
    # print(f"strides {*strides, *mat1.stride(), *output.stride()}")
    # print(f"input args {output, coefs, mat1, M, P, Q, L, *strides, *mat1.stride(), *output.stride(), seed, scale_bias, int(fused_bias)}")
    rt_mm_kernel[grid](output, coefs, mat1, M, P, Q, L, *strides, *mat1.stride(), *output.stride(), seed, scale_bias, int(fused_bias))
    return output


@triton.jit
def rt_mm_grouped_kernel(out_ptr,    # ptr to output vector
                coef_ptr,  # ptr to the coefficients vector
                m1_ptr, # ptr to the other other tensor mat1
                buffer_ptr, # ptr to a buffer vector that is (GROUP_SIZE * BLOCK_SIZE_L, BLOCK_SIZE_Q)
                M,   # size of the RT int dim
                P, Q,	# size of the RT (P, Q)
                L,	# size of the other tensor mat1 (Q, L)
                stride_rt_P, stride_rt_Q, # strides of RT
                stride_m1_L, stride_m1_P,	# strides of other tensor mat1
                stride_out_L, stride_out_Q,
                seed,   # seed for the generated tensor
                #TODO: scale bias fused
                scale_bias_ptr, RAND_BIAS: tl.constexpr,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_L: tl.constexpr, GROUP_SIZE: tl.constexpr):
    """
    Fused kernel computing the matmul eqn: mat1 @ RT = out | (L, P) @ (P, Q) = (L, Q)
    
    This one where we explicitly maintain a buffer_ptr vector that is used to tl.store() intermediate representations of RT that is computed. SHOULD be in L2, slightly slower than compared to full materialization? But this is
    an issue with triton, where we cannot explicitly manage the memory layout and cannot materialize buffer vector in shared memory.
    
    The cost of materializing a column of the RT is the bitshift squirrel5 kernel and a dot product against a coef vector which should already be in L1. It should be confirmed and benched the perf of L1 materialization vs loading
    an already materialized variant of the column from L2. 
    Here we allocate certain pids to specific columns of the RT and maintain the pid persistently to keep the column in L1 cache, then load the required row from mat1. But this results in very bad mem access pattern?
    Maybe each pid assigned to 1 col of RT, materialize it in L1, then load first row of mat1, then second row, sequentially, until run out of rows, then move onto the next block of columns and repeat per each row?
    This would write consecutive indices of out matrix too, first row of out mat, then second row etc. for coalesced writes. This means that every pid at the same iterations would mem read the same row of mat1 and the
    mem read would be broadcasted to all the L1 caches making it faster.
    
    In torch the usual memory format for tensors is row-major, which is slightly more inefficient when accessing elements for mat2 (hence some ops
    pass in mat2.t() like linear layer). This is completely fine if mat2 is an RT, since in that case we can compute the output in row-major without
    loading the whole mat2 in memory, instead of having to group-order the output, increasing chances for coalesced writes.
    
    For each elem in output C, we load the corresponding mat1 row (1,P) into SRAM, then compute the respective noise matrix for a column of B (P, M).
    The noise matrix materialized via squirrel5_gen_2d requires offs_n, which is the indices of P modified by the strides.
    Then run a dot product along P-dim against the coef vector (M,) to get a vector of the column B (P,) which we then dot product with mat1 row (1,P).
    The two dot products can actually be fused, but might be more efficient to let compiler handle that.
    """
    # identifies consecutive part of the output vector we are writing to (can be across L-dim)
    # different pids of L will regenerate the col, so there will be SPLIT_L number of times that the RT col is recomputed,
    # which can further be split into BLOCK_SIZE_L for loop within a pid. But this is for future work, the idea is that rows in mat1
    # remain in L2 cache and are not completely flushed. This should follow the grouped ordering in triton docs? Won't work because
    # multiple warps/blocks will attempt to read the same row of mat1 at the same time, and broadcasts cannot be serialised to multiple
    # warps. This is probably a big bottleneck. However we want to minimize the number of "reaccesses" to the RT
    pid_l, pid_q = tl.program_id(0), tl.program_id(1)
    num_pid_l, num_pid_q = tl.num_programs(0), tl.num_programs(1)
    # pid_l, pid_q = tl.swizzle2d(pid_l, pid_q, num_pid_l, num_pid_q, GROUP_SIZE)
    pid_q, pid_l = tl.swizzle2d(pid_q, pid_l, num_pid_q, num_pid_l, GROUP_SIZE)
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = offs_l < L
    offs_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    mask_q = offs_q < Q
    
    # split along P dim
    pid_p = tl.program_id(2)
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    mask_p = offs_p < P
    
    col = tl.zeros([BLOCK_SIZE_P * BLOCK_SIZE_Q], dtype=tl.float32)
    offs_n = tl.ravel(offs_p[:, None] * stride_rt_P + offs_q[None, :] * stride_rt_Q)	# (BLOCK_SIZE_P, BLOCK_SIZE_Q)
    # squirrel5 noise function to generate a block of the column, that block being some consecutive BLOCK_SIZE_P values
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mangledBits = squirrel5_gen_2d(offs_m, offs_n, seed)    # (BLOCK_SIZE_P * BLOCK_SIZE_Q, BLOCK_SIZE_M)
        coefs = tl.load(coef_ptr + offs_m, mask=offs_m < M, other=0.0)[None, :]    # (1, BLOCK_SIZE_M)
        col += tl.sum(mangledBits * coefs, axis=1)  # (BLOCK_SIZE_P * BLOCK_SIZE_Q)
    #TODO: i think there has to be masking for the col P,Q dims, but can't figure it out after reshape? might need tl.where
    col = tl.where(mask_p[:, None] & mask_q[None, :], tl.reshape(col, (BLOCK_SIZE_P, BLOCK_SIZE_Q), can_reorder=False), 0.0)
    # col = tl.reshape(col, (BLOCK_SIZE_P, BLOCK_SIZE_Q), can_reorder=False)
    
    # multiply the BLOCK_SIZE_P col values with corresponding BLOCK_SIZE_P row values along the same P offset
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_l = tl.max_contiguous(tl.multiple_of(offs_l, BLOCK_SIZE_L), BLOCK_SIZE_L)
    row_ptrs = m1_ptr + offs_l[:, None] * stride_m1_L + offs_p[None, :] * stride_m1_P
    out_ptrs = out_ptr + offs_l[:, None] * stride_out_L + offs_q[None, :] * stride_out_Q
    # it seems that global memory loads are not broadcasted to different warps if they all request the same row
    #TODO: the loading scheme for rows seems to be uncoalesced? 1 byte per sector used 
    row = tl.load(row_ptrs, mask=mask_l[:, None] & mask_p[None, :], other=0.0)	# (BLOCK_SIZE_L, BLOCK_SIZE_P)
    dot_prod = tl.dot(row, col, allow_tf32=False)
    mask_out = mask_l[:, None] & mask_q[None, :]
    tl.atomic_add(out_ptrs, dot_prod, mask=mask_out, sem="relaxed")
    # mask_out = mask_l[:, None] & mask_q[None, :]
    # dot_prod = tl.dot(row, col, allow_tf32=False)
    # # if allow_tf32 is True the fp32xfp32 dot product will be pretty significantly numerically different from correct values: https://github.com/triton-lang/triton/issues/4574
    # tl.atomic_add(out_ptrs, dot_prod, mask=mask_out, sem="relaxed")
    # interleaved. but idk why this helps at all, the interleave is across row 
    # acc = tl.zeros(BLOCK_SIZE_L * SPLIT_L, BLOCK_SIZE_Q)
    # for l in range(0, tl.cdiv(L, BLOCK_SIZE_L*SPLIT_L)):
    #     # mask_l = offs_l < GROUP_SIZE_L - l * BLOCK_SIZE_L
    #     mask_l = offs_l < L - l * BLOCK_SIZE_L*SPLIT_L
    #     # it seems that global memory loads are not broadcasted to different warps if they all request the same row
    #     #TODO: the loading scheme for rows seems to be uncoalesced? 1 byte per sector used 
    #     row = tl.load(row_ptrs, mask=mask_l[:, None] & mask_p[None, :], other=0.0)	# (BLOCK_SIZE_L, BLOCK_SIZE_P)
        
    #     mask_out = mask_l[:, None] & mask_q[None, :]
    #     # if allow_tf32 is True the fp32xfp32 dot product will be pretty significantly numerically different from correct values: https://github.com/triton-lang/triton/issues/4574
    #     tl.atomic_add(out_ptrs, dot_prod, mask=mask_out, sem="relaxed")
    #     # advance to the next row
    #     row_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_m1_L
    #     out_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_out_L
    
if int(os.environ["AUTOTUNE"]):
    rt_mm_grouped_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_M": [32],
                                                "BLOCK_SIZE_P": [32, 64],
                                                "BLOCK_SIZE_Q": [32, 64],
                                                "BLOCK_SIZE_L": [32, 64],
                                                "GROUP_SIZE": [2, 3, 4]}, None, None, num_warps=4, pre_hook=_zero_output_outptr),
                                           key=['M', 'P', 'Q', 'L'])(rt_mm_grouped_kernel)
else:
    rt_mm_grouped_kernel.add_pre_run_hook(_zero_output_outptr)

def rt_mm_grouped(mat1: torch.Tensor, coefs: torch.Tensor, size: list[int], strides: list[int], seed: int, scale_bias: torch.Tensor, fused_bias: bool) -> torch.Tensor:
    M = coefs.size(0)
    P, Q = size
    L = mat1.size(0)
    assert mat1.size(1) == P, f"mm shape mismatch, received mat1 shape {mat1.size()} and rt shape {size}"
    assert not (fused_bias and scale_bias is None), "scale_bias not initialized but fused_bias is True"
    output = torch.empty(size=(L, Q), device=coefs.device)
    grid = lambda meta: ( triton.cdiv(L, meta['BLOCK_SIZE_L']), triton.cdiv(Q, meta['BLOCK_SIZE_Q']), triton.cdiv(P, meta['BLOCK_SIZE_P']) )
    print(f"strides {*strides, *mat1.stride(), *output.stride()}")
    # print(f"input args {output, coefs, mat1, M, P, Q, L, *strides, *mat1.stride(), *output.stride(), seed, scale_bias, int(fused_bias)}")
    rt_mm_grouped_kernel[grid](output, coefs, mat1, M, P, Q, L, *strides, *mat1.stride(), *output.stride(), seed, scale_bias, int(fused_bias))
    return output

class fakeMMFunction(Function):
    """
    - func: mm(Tensor self, Tensor mat2) -> Tensor
    Intermediate wrapper function while I debug the fused matmul kernel.
    """
    @staticmethod
    def forward(ctx, self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(self, mat2)
        # print(f"fakeMM fwd called on {self} {mat2}")
        return self @ mat2

    @staticmethod
    def backward(ctx, grad):
        self, mat2 = ctx.saved_tensors
        print(f"fakeMM bwd called on {self} {mat2} with grad {grad}")
        dself = grad @ mat2.t()
        dmat2 = grad @ self.t()
        return dself, dmat2

def fakeMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return fakeMMFunction.apply(mat1, mat2)


class RandumbTensor(torch.Tensor):
    """
    This is a structured tensor representation that is constructed via the fused matmul of two tensors, tensor A that is generated 
    via prng function and tensor B, a trainable coefficient tensor. Tensor A has shape (n, int_dim). Tensor B has shape (int_dim,).
    
    This custom tensor subclass should be usable anywhere, however when utilising it inside nn.Module, remember to register
    the _coef tensor as a parameter of the module, and the RandumbTensor itself as a buffer via `register_buffer` for any
    optimizers and stuff like `.to(device)` to work.
    
    Do note that due to the nature of this tensor, device is expected to be the same device as _coef tensor, and when used should be 'cuda:{whatever_gpu}'.
    Always instantiate a RandumbTensor via the RandumbTensorConstructor, this ensures that the gradients will be properly backpropagated
    to the coeficient tensor. 
    
    There is a slightly complicated "lazy" view materialization, it works by intercepting any torch_dispath to view/transpose/etc. ops 
    (defined by OVERRIDE_VIEW_OPS) called on the RT, and returns another RT instance with correctly updated shape and stride information 
    (used in materialization kernel). The shape and stride information is collected by simulating the view op through a faketensor, and 
    if any view ops that are incompatible with the current RT's size and stride is called, it will throw an error through faketensormode.
    
    While coef will work with bfloat16, float16 and float32, and the variable of self.dtype follows the coef.dtype, take note that
    the final output of the materialization kernel will be upcasted to float32. If .to() is utilized to cast the output dtype, then it will be 
    treated as view_ops under _to_copy().
    
    Integration with torch compile is not supported, there is no guarantee that torch compile will not create a model with a memory leak
    due to it compiling a graph that holds references to _tensor - this is the case for the modded-nanogpt implementation. 
    However torch compile might still work for simpler models such as mnist, so use at your own risk.
    """
    
    @staticmethod
    def __new__(cls, data: torch.Tensor, seed: int, shape: Union[tuple, torch.Size], scale_bias: torch.Tensor, fused_bias: bool, strides: tuple[int] = None):
        # Creating the wrapper will generally detach tensors from the autograd graph, ensure that the 
        # input does not require grad or not created inside autograd context (should be taken care of by constructor Function)
        assert not data.requires_grad or not torch.is_grad_enabled()
        kwargs = {}
        if strides is not None: kwargs["strides"] = strides
        return torch.Tensor._make_wrapper_subclass(cls, torch.Size(shape), device=data.device, dtype=data.dtype, requires_grad=False, **kwargs)

    def __init__(self, data: torch.Tensor, seed: int, shape: Union[tuple, torch.Size], scale_bias: torch.Tensor, fused_bias: bool, strides: tuple[int] = None):
        """
        Args:
            data (torch.Tensor): If `int`, represents intermediate/intrinsic dimension, and random coef
                tensor will be generated. Can be nn.Parameter (expected for usage in Layers).
            seed (int): Seed for the prng generator
            shape (Union[tuple, torch.Size]): Final tensor output shape
            dtype (torch.dtype): dtype
            device (str, optional): device variable cannot actually be properly updated with _C.TensorBase class. 
                This is instead only used for initialising _coef device, the class's device type will always follow _coef.
                Defaults to 'cpu'.
            requires_grad (bool, optional): Requires grad. Defaults to False.
        """
        self.seed = seed
        self.scale_bias = scale_bias
        self.fused_bias = fused_bias
        
        # initialise coefficient tensor from given data
        self._coef = data
        size = self._coef.size()
        assert len(size) == 1, f"Coefficient tensor must be 1D but received tensor of shape {self.size}"
        self.int_dim = size[0]
        
        # final output tensor will be materialised in this reference whenever necessary
        self._tensor: torch.Tensor = None
        # shape of the prng generated tensor (that should never be fully materialised)
        self.noise_matrix_shape = torch.Size((self.shape.numel(), self.int_dim))
    
    def __repr__(self):
        with torch._C._DisableTorchDispatch():
            return f"RandumbTensor(seed={self.seed}, int_dim={self.int_dim}, shape={self.shape}, stride={self.stride()}, noise_matrix_shape={self.noise_matrix_shape}, dtype={self.dtype}, device='{self.device}')"
    
    def get_materialized(self) -> torch.Tensor:
        # helper function to materialize the full matrix, should only be used for debugging tbh, for all intents and purposes other than
        # printing the output tensor, just calling rt should suffice as the torch_dispatch override should materialize the matrix anyway
        self.materialize()
        ref = self._tensor
        self.dematerialize()
        return ref
    
    def get_noise_matrix(self) -> torch.Tensor:
        # helper function to materialize the standalone noise matrix, should only be used for debugging
        true_noise_mat = squirrel5_generate_2d(*self.noise_matrix_shape, self.seed)
        return true_noise_mat
    
    def print_memory(self):
        # helper function to describe memory usage of this tensor
        # calculations in MiB
        noise_mem = self.element_size() * self.noise_matrix_shape.numel() / 2**20
        coeff_mem = self._coef.element_size() * self._coef.numel() / 2**20
        out_mem = self.element_size() * self.shape.numel() / 2**20
        total_mem = noise_mem + coeff_mem + out_mem
        print(f"unmaterialized noise matrix: {noise_mem}MiB\ncoefficient matrix: {coeff_mem}MiB\noutput matrix: {out_mem}MiB")
        print(f"% 'saved' memory: {noise_mem / total_mem * 100}%")
        return
    
    # disable torch compile as it causes some extremely funky bugs where _tensor does not exist
    @torch.compiler.disable(recursive=False)
    def materialize(self):
        # This should be outside the autograd computation graph already if ever called, but just in case we wrap it in torch.no_grad
        with torch.no_grad():
            # materializes the tensor, then reshape it appropriately. 
            self._tensor = materialize_fwd(self._coef, self.size(), self.stride(), self.seed, self.scale_bias, self.fused_bias).to(self.dtype)

    def dematerialize(self):
        # this function should free _tensor from memory. We can safely delete the entire tensor because the autograd computation is tracked manually by the
        # wrapper creation Function.
        
        # if memory leak suspected, uncomment. Otherwise keep commented as it causes torch.compile to graph break (if you are using it)
        """
        if sys.getrefcount(self._tensor) > 2: 
            # ref count should be 2 - self._tensor variable itself and the globals() dict. be aware that torch.compile'd functions that use ._tensor can
            # unintentionally maintain references to it (i.e. in torch_dispatch) or during dynamo JIT compilation.
            warnings.warn(f"Dematerialize memory leak: ref count of _tensor {sys.getrefcount(self._tensor)}>2, some reference to _tensor still exists.\n{referrers.get_referrer_graph(self._tensor)}")
        """
        del self._tensor
        return
    
    __torch_function__ = torch._C._disabled_torch_function_impl
    # list of view ops that we intercept in torch_dispatch, should be easy to add support for any listed https://pytorch.org/docs/stable/tensor_view.html
    # full list of aten ops https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
    OVERRIDE_VIEW_OPS = [torch.ops.aten.view.default, torch.ops.aten.transpose.int, torch.ops.aten._to_copy.default, torch.ops.aten.t.default, torch.ops.aten.permute.default]
    # As long as all the view ops performed on the RT are compatible with tensor's size and stride, we can simply alter the shape and stride accordingly and the 
    # materialize_fwd accepting strides will work its magic. This means we also don't need to store lists of view ops, while faketensor will still throw an error
    # if an illegal view_op is performed on a possibly non-contiguous tensor (i.e. Error like "view size is not compatible ... Call .contiguous() before .view().")
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # keep track of all materialized tensors
        materialized_tensors: list[RandumbTensor] = []
        def _materialize(e):
            if isinstance(e, RandumbTensor):
                e.materialize()
                materialized_tensors.append(e)
                return e._tensor
            else: 
                return e
            
        # perform the operation with materialized tensors
        if func in cls.OVERRIDE_VIEW_OPS:
            instance: RandumbTensor = args[0]
            with torch.no_grad():
                # use faketensor to compute the final output shape, because we cannot change tensor.shape attribute once created, and if the output 
                # shape doesn't match the grad that is returned in the bwd pass, then pytorch will throw an error.
                fake_mode = detect_fake_mode(args)
                if fake_mode is None: fake_mode = FakeTensorMode()
                with fake_mode:
                    fake = torch.empty_strided(instance.shape, instance.stride())
                    fake = func(fake, *args[1:], **kwargs)
                rt = RandumbTensor(instance._coef, instance.seed, fake.shape, instance.scale_bias, instance.fused_bias, fake.stride())
                del fake
            return rt
        elif func is torch.ops.aten.mm.default:
            # print("RT detected aten.mm op, overriding with custom mm args")
            # print(args)
            # unwrap, because after this is called, inside fakeMM it again calls a matmul, which would otherwise redispatch here, and cause an infinite loop
            ret = fakeMM(*tree_map(_materialize, args), **tree_map(_materialize, kwargs))
        else:
            ret = func(*tree_map(_materialize, args), **tree_map(_materialize, kwargs))
            
        # dematerialize all previously materialized tensors
        for t in materialized_tensors:
            t.dematerialize()
            
        return ret
    
    # TODO: implement indexing and slicing like https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/uint4_tensor.py#L89
    

class RandumbTensorConstructor(Function):
    """
    A differentiable function that constructs and returns a RandumbTensor given the coefs, seed and shape.
    This function manually handles the bwd function backpropping gradients to the coef tensor.
    Always use this function to construct a RandumbTensor, and do not construct one directly.
    """
    @staticmethod
    def forward(ctx, coefs: torch.Tensor, seed: int, shape: Iterable, scale_bias: torch.Tensor, fused_bias: bool) -> RandumbTensor:
        if type(coefs) is torch.Tensor or torch.nn.Parameter:
            assert len(coefs.size()) == 1, f"Coefficient tensor must be 1D but received tensor of shape {coefs.size()}"
            out = RandumbTensor(coefs, seed, shape, scale_bias, fused_bias)

            ctx.dcoefs_size = coefs.numel()
            ctx.seed = seed
            ctx.dtype = out.dtype
            ctx.fused_bias = fused_bias
            return out
        else:
            raise AssertionError(f"Expected coefs to be of type torch.Tensor, found {type(coefs)} instead.")

    @staticmethod
    def backward(ctx, grad):
        # flatten the grad back to same shape as materialize bwd output
        # .flatten() instead of .view() as no guarantee the bwd tensor is contiguous
        print("RT constructor bwd called")
        grad = grad.flatten()
        # run the actual bwd
        dcoefs = materialize_bwd(grad, ctx.dcoefs_size, ctx.seed).to(ctx.dtype)
        dscale_bias = None
        if ctx.fused_bias: dscale_bias = scale_bias_bwd(grad, ctx.seed)
        # only coefs needs grad, the rest are non-differentiable. function signature of fwd: coefs, size, seed, scale_bias, fused_bias
        return dcoefs, None, None, dscale_bias, None

# alias for the constructor, use this as `rt = CreateRandumbTensor(...)`
def CreateRandumbTensor(coefs: torch.Tensor, seed: int, shape: Iterable, scale_bias: torch.Tensor = None, fused_bias: bool = False) -> RandumbTensor:
    return RandumbTensorConstructor.apply(coefs, seed, shape, scale_bias, fused_bias)


# triton kernel implementation for the embedding layer
@triton.jit
def embed_fwd_kernel(out_ptr,   # ptr to output vector
                coef_ptr,   # ptr to the coefficients vector
                idx_ptr,    # ptr to the row indices vector
                rowsize,    # row size. should be stride[0] of the 2d embedding matrix
                M, idx_size, # sizes of the vectors
                seed,   # seed for the generated tensor
                stride_out_row, # stride[0] of the output vector
                BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_IDX: tl.constexpr):
    """
    Fused kernel utilizing squirrel5 noise function to materialize specific idx_ptr rows of a RandumbTensor acting as a 2d embedding matrix
    The out_ptr is the output vector of shape (idx_size, rowsize)
    The coef_ptr is the coefficient vector of shape (M,)
    This function should launch using a 2d grid of BLOCK_SIZE_IDX, BLOCK_SIZE_N. Each pid is responsible for a BLOCK_SIZE_IDX num of indices and
    BLOCK_SIZE_ROW chunk of the output matrix indexed by idx_ptr[pid]. The pid lazily materializes a BLOCK_SIZE_ROW sequential chunk of the 
    (rowsize, M) subset of the noise matrix (indexed by idx_ptr) and matmuls it with the coef matrix in BLOCK_SIZE_M chunks.
    It might be a bit inefficient compared to the alternative of launching one pid per index, but if you can oneshot the rowsize-dim computation or
    allocating rowsize'd chunk doesn't completely oom the GPU, then we might as well allocate the blocks for other indices as well.
    """
    pid_idx = tl.program_id(0)
    pid_row = tl.program_id(1)
    
    # locates correct block of indices to load
    offs_idx = pid_idx * BLOCK_SIZE_IDX + tl.arange(0, BLOCK_SIZE_IDX)
    mask_idx = offs_idx < idx_size
    idxs = tl.load(idx_ptr + offs_idx, mask_idx)
    
    # usually non-contiguous memory reads would be extremely expensive, but since the noise matrix doesn't actually exist in memory, we can materialize
    # arbitrary non-contiguous row indices. So we compute [BLOCK_SIZE_IDX, BLOCK_SIZE_ROW], each entry in BLOCK_SIZE_IDX is the ptr locating the start of
    # the row, then BLOCK_SIZE_ROW locates the specific chunk of the row that we are computing on. Then we flatten everything since squirrel5_gen_2d accepts 1D input.
    offs_row = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offs_n = tl.ravel( (idxs * rowsize)[:, None] + offs_row[None, :] )
     
    # iterate along M-dim. These aren't "true" blocks in the sense that computation for blocks is split between pids, all chunks of computation here
    # are done in the same pid/block, this is more like "phases" within a single block, beacuse we require value of the entire vector to compute the dot product.
    accumulator = tl.zeros((BLOCK_SIZE_IDX, BLOCK_SIZE_ROW), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        
        # squirrel5 noise function
        mangledBits = squirrel5_gen_2d(offs_m, offs_n, seed)   # [BLOCK_SIZE_IDX * BLOCK_SIZE_ROW, BLOCK_SIZE_M]
        
        # mask off mangledBits that are > M, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_M = offs_m < M
        # load the subgroup of coefficients we are processing
        coefs = tl.load(coef_ptr + offs_m, mask=mask_M)[None, :]  # [1, BLOCK_SIZE_M]
        
        # dot product via broadcast of elementwise multiplication of coeffs across N-dim of materialized noise matrix
        # followed by summation along M-dim
        accumulator += tl.sum(mangledBits * coefs, axis=1).reshape(BLOCK_SIZE_IDX, BLOCK_SIZE_ROW, can_reorder=False)  # [BLOCK_SIZE_IDX, BLOCK_SIZE_ROW]
        
    # locates the ptr for the correct idx entries
    offs_idx = (offs_idx * stride_out_row)[:, None]
    # locates ptrs for correct BLOCK of row that this pid computed
    offs_row = offs_row[None, :]
    out_ptrs = out_ptr + offs_idx + offs_row
    mask = (offs_idx < idx_size * rowsize) & (offs_row < rowsize)
    tl.store(out_ptrs, accumulator, mask=mask)

@triton.jit
def embed_bwd_kernel(dcoef_ptr,    # ptr to dcoef vector
                dout_ptr,  # ptr to the dout vector
                idx_ptr,    # ptr to the row indices vector
                rowsize,    # row size. should be stride[0] of the 2d embedding matrix
                M, idx_size, # sizes of the vectors
                seed,   # seed for the generated tensor
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, SPLIT_N: tl.constexpr):
    """
    Backward pass of the embedding kernel. Implements split-n coalesced loads.
    """
    # identifies correct memory address of the part of the dcoef vector we are writing to
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
     
    # start at first block to load
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)    
    # these are just compiler hints, note multiple_of documentation is wrong, ref: https://github.com/triton-lang/triton/issues/1324
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # size of each coalesced load
    LOAD_SIZE = BLOCK_SIZE_N * SPLIT_N
    N = idx_size * rowsize
    # iterate along N-dim (of the "transposed" noise matrix)
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, LOAD_SIZE)):
        # should most often be a block of constant values denoting the offs for particular idx of the block, but at the boundary
        # between blocks if rowsize % BLOCK_SIZE_N != 0 there will be different idxs.
        offs_idx = offs_n // rowsize
        idx = tl.load(idx_ptr + offs_idx, mask=offs_idx < idx_size)
        # squirrel5 noise function, the correct offs_n should be idx + offs relative for that idx, not offs_n from the dout_ptr, hence % rowsize
        mangledBits = squirrel5_gen_2d_T(offs_m, idx * rowsize + (offs_n % rowsize), seed)   # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        
        # mask off mangledBits that are > N, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_N = offs_n < N 
        # load the subgroup of dout we are processing
        dout = tl.load(dout_ptr + offs_n, mask=mask_N)[None, :]  # [1, BLOCK_SIZE_N]
        
        # dot product via broadcast of elementwise multiplication of dout across M-dim of materialized noise matrix
        # followed by summation along N-dim
        accumulator += tl.sum(mangledBits * dout, axis=1) # [BLOCK_SIZE_M,]
        # jump to next block to load
        offs_n += LOAD_SIZE
        
    accumulator = accumulator.to(dcoef_ptr.dtype.element_ty)
    tl.atomic_add(dcoef_ptr + offs_m, accumulator, mask=mask_m, sem="relaxed")

if int(os.environ["AUTOTUNE"]):
    embed_fwd_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_ROW": [256, 1024], 
                                                                  "BLOCK_SIZE_M": [16, 64],  
                                                                  "BLOCK_SIZE_IDX": [16, 64]}, None, None, num_warps=4), 
                                       key=['rowsize', 'M', 'idx_size'])(embed_fwd_kernel)
    embed_bwd_kernel = triton.autotune(configs=construct_configs({"BLOCK_SIZE_N": [32, 128, 1024], "BLOCK_SIZE_M": [8, 16, 32], "SPLIT_N": [4, 8, 16]},
                                                                 None, None, num_warps=4, pre_hook=_zero_output_coef), 
                                       key=['rowsize', 'M', 'idx_size'])(embed_bwd_kernel)
else:
    embed_bwd_kernel.add_pre_run_hook(_zero_output_coef)

@triton_op("rten::embed_fwd", mutates_args={})
def embed_fwd(idx: torch.Tensor, coefs: torch.Tensor, rowsize: int, seed: int) -> torch.Tensor:
    output = torch.empty(size=(*idx.shape, rowsize), device=coefs.device)
    grid = lambda meta: ( triton.cdiv(idx.numel(), meta['BLOCK_SIZE_IDX']), triton.cdiv(rowsize, meta['BLOCK_SIZE_ROW']), )
    wrap_triton(embed_fwd_kernel)[grid](output, coefs, idx, rowsize, coefs.numel(), idx.numel(), seed, output.stride(-2))
    return output

@triton_op("rten::embed_bwd", mutates_args={})
def embed_bwd(idx: torch.Tensor, dout: torch.Tensor, dcoefs_size: int, rowsize: int, seed: int) -> torch.Tensor:
    # note to self: should we enforce dout to be a flattened tensor? it doesn't seem to be necessary
    dcoefs = torch.empty(size=(dcoefs_size,), device=dout.device)
    grid = lambda meta: (triton.cdiv(dcoefs_size, meta['BLOCK_SIZE_M']), meta['SPLIT_N']) 
    wrap_triton(embed_bwd_kernel)[grid](dcoefs, dout, idx, rowsize, dcoefs_size, idx.numel(), seed)
    return dcoefs

def _embed_backward(ctx, grad):
    idx = ctx.saved_tensors[0]
    dcoefs = embed_bwd(idx, grad, ctx.dcoefs_size, ctx.rowsize, ctx.seed)
    # only coefs needs grad, the rest are non-differentiable. function signature of fwd: idx, coefs, rowsize, seed
    return None, dcoefs, None, None

def _embed_setup_context(ctx, inputs, output):
    idx, coefs, rowsize, seed = inputs
    ctx.save_for_backward(idx)
    ctx.dcoefs_size = coefs.numel()
    ctx.rowsize = rowsize
    ctx.seed = seed

embed_fwd.register_autograd(_embed_backward, setup_context=_embed_setup_context)

@embed_fwd.register_kernel("cpu")
def _embed_fwd(idx: torch.Tensor, coefs: torch.Tensor, rowsize: int, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of embed_fwd called.")

@embed_bwd.register_kernel("cpu")
def _embed_bwd(dout: torch.Tensor, dcoefs_size: int, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of embed_bwd called.")
