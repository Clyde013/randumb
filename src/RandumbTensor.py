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
"""

from typing import Union
from collections.abc import Iterable

import torch
from torch.utils._pytree import tree_map, tree_map_only
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton, custom_op

from torch.autograd import Function

import triton
import triton.language as tl

from torch.profiler import profile, record_function

# the following is a standalone kernel implementation, used mainly for crosschecking correctness of the more complex implementations
@triton.jit
def squirrel5_kernel(x_ptr,	# pointer to output vector
                     n,	# size of the vector
                     BLOCK_SIZE: tl.constexpr,
                     seed,	# seed for the tensor being generated
                     pos,	# when generating 2d matrix by each column, this is the column number
                     ):

    # squirrel noise constants
    SQ5_BIT_NOISE1: tl.constexpr = 0xd2a80a3f	# 11010010101010000000101000111111
    SQ5_BIT_NOISE2: tl.constexpr = 0xa884f197	# 10101000100001001111000110010111
    SQ5_BIT_NOISE3: tl.constexpr = 0x6C736F4B # 01101100011100110110111101001011
    SQ5_BIT_NOISE4: tl.constexpr = 0xB79F3ABB	# 10110111100111110011101010111011
    SQ5_BIT_NOISE5: tl.constexpr = 0x1b56c4f5	# 00011011010101101100010011110101
    PRIME_NUMBER: tl.constexpr = 198491317 # Large prime number with non-boring bits
    
    # identifies correct memory address of the part of the output vector we are writing to
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    
    # squirrel5 noise function
    # use PRIME_NUMBER to mix pos into single int for the noise function
    mangledBits = offs.to(tl.int32) + (PRIME_NUMBER * pos)
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
    
    tl.store(x_ptr + offs, mangledBits, mask=mask)
     
    #print_if(f'pid = {pid} | mangledBits = {mangledBits} | ONE_OVER_MAX_INT = {ONE_OVER_MAX_INT}', '')

def squirrel5_generate(row_num, size=300, BLOCK_SIZE=1024, seed=1337):
    output = torch.empty(size=(size,), device='cuda')
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    squirrel5_kernel[grid](output, n_elements, BLOCK_SIZE=BLOCK_SIZE, seed=seed, pos=row_num)
    return output




@triton.jit
def squirrel5_gen(cols: tl.tensor, rows: tl.tensor, seed):
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
    
    mangledBits = (PRIME_NUMBER * rows)[:, None] + cols.to(tl.int32)[None, :]
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
def squirrel5_gen_T(cols: tl.tensor, rows: tl.tensor, seed):
    """    
    Transposed variation of the squirrel5_gen algorithm, utilised in the backward pass. Note the transposition of the n, m matrices in the rows & cols definition below.
    
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
    
    mangledBits = (PRIME_NUMBER * rows)[None, :] + cols.to(tl.int32)[:, None]
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


@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_warps=4),
  ],
  key=['N', 'M']
)
@triton.jit
def materialize_fwd_kernel(out_ptr,    # ptr to output vector
                coef_ptr,  # ptr to the coefficients vector
                N, M,   # sizes of the vectors
                seed,   # seed for the generated tensor
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    Fused kernel utilizing squirrel5 noise function to materialize a (N,) sized output vector (that is later arbitrarily reshaped)
    The out_ptr is the output vector of shape (N,)
    The coeff_ptr is the coefficient vector of shape (M,)
    The noise function should lazily materialize the (N, M) matrix row by row (along N dim), each row getting fused dot product'd with
    the entire coef matrix, which should result in cache hits every time because the coef matrix should be perma loaded into SRAM.
    The resulting equation is:    noise @ coef = out
    """
    # identifies correct memory address of the part of the output vector we are writing to
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N
    
    # iterate along M-dim. These aren't "true" blocks in the sense that computation for blocks is split between pids, all chunks of computation here
    # are done in the same pid/block, this is more like "phases" within a single block, beacuse we require value of the entire vector to compute the dot product.
    accumulator = tl.zeros((BLOCK_SIZE_N, 1), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        
        # squirrel5 noise function
        mangledBits = squirrel5_gen(offs_m, offs_n, seed)   # [BLOCK_SIZE_N, BLOCK_SIZE_M]
        
        # mask off mangledBits that are > M, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_M = offs_m < M
        # load the subgroup of coefficients we are processing
        coefs = tl.load(coef_ptr + offs_m, mask=mask_M)[None, :]  # [1, BLOCK_SIZE_M]
        
        # dot product via broadcast of elementwise multiplication of coeffs across N-dim of materialized noise matrix
        # followed by summation along M-dim
        accumulator += tl.sum(mangledBits * coefs, axis=1)[:, None]  # [1, BLOCK_SIZE_N]
        
    tl.store(out_ptr + offs_n[:, None], accumulator, mask=mask[:, None])


@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_warps=4),
  ],
  key=['N', 'M']
)
@triton.jit
def materialize_bwd_kernel(dcoef_ptr,    # ptr to dcoef vector
                dout_ptr,  # ptr to the dout vector
                N, M,   # sizes of the vectors
                seed,   # seed for the generated tensor
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    Backwards pass of the above materialization kernel. This one will compute dcoef via dout and generating the transposed noise matrix.
    Since the noise matrix is transposed, the dims of it are (m, n) instead.
    given the eqn: noise @ coef = out |   (n, m) @ (m, 1) = (n, 1)
    dcoef = noise.T @ dout    |   (m, 1) = (m, n) @ (n, 1)
    it should be possible to generate the transpose of the noise matrix and fuse the matmul somehow
    making the backward pass for this function almost the exact same time complexity as the forward pass.
    """
    # identifies correct memory address of the part of the dcoef vector we are writing to
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offs_m < M
    
    # iterate along N-dim (of the "transposed" noise matrix)
    accumulator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # squirrel5 noise function
        mangledBits = squirrel5_gen_T(offs_m, offs_n, seed)   # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        
        # mask off mangledBits that are > N, otherwise this will affect the final dot product
        # we do this by masking off the respective coef that would elementwise multiply with the
        # out of bounds mangledBits, thereby masking those out as well
        mask_N = offs_n < N
        # load the subgroup of dout we are processing
        dout = tl.load(dout_ptr + offs_n, mask=mask_N)[None, :]  # [1, BLOCK_SIZE_N]
        
        # dot product via broadcast of elementwise multiplication of dout across M-dim of materialized noise matrix
        # followed by summation along N-dim
        accumulator += tl.sum(mangledBits * dout, axis=1)[:, None]  # [1, BLOCK_SIZE_M]
        
    tl.store(dcoef_ptr + offs_m[:, None], accumulator, mask=mask[:, None])


@triton_op("randumblib::materialize_fwd", mutates_args={})
def materialize_fwd(coefs: torch.Tensor, size: int, seed: int) -> torch.Tensor:
    output = torch.empty(size=(size,), device=coefs.device)
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE_N']), )
    wrap_triton(materialize_fwd_kernel)[grid](output, coefs, size, coefs.numel(), seed)
    return output

@triton_op("randumblib::materialize_bwd", mutates_args={})
def materialize_bwd(dout: torch.Tensor, dcoefs_size: int, seed: int) -> torch.Tensor:
    # note to self: should we enforce dout to be a flattened tensor? it doesn't seem to be necessary
    dcoefs = torch.empty(size=(dcoefs_size,), device=dout.device)
    grid = lambda meta: (triton.cdiv(dcoefs_size, meta['BLOCK_SIZE_M']), )
    wrap_triton(materialize_bwd_kernel)[grid](dcoefs, dout, dout.numel(), dcoefs_size, seed)
    return dcoefs

def backward(ctx, grad):
    dcoefs = materialize_bwd(grad, ctx.dcoefs_size, ctx.seed)
    # only coefs needs grad, the rest are non-differentiable. function signature of fwd: coefs, size, seed, device
    return dcoefs, None, None, None

def setup_context(ctx, inputs, output):
    coefs, size, seed = inputs
    ctx.dcoefs_size = coefs.numel()
    ctx.seed = seed

materialize_fwd.register_autograd(backward, setup_context=setup_context)

@materialize_fwd.register_kernel("cpu")
def _fwd(coefs: torch.Tensor, size: int, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of materialize_fwd called.")

@materialize_bwd.register_kernel("cpu")
def _bwd(dout: torch.Tensor, dcoefs_size: int, seed: int):
    raise NotImplementedError("Unsupported CPU implementation of materialize_fwd called.")


class RandumbTensor(torch.Tensor):
    """
    This is a structured tensor representation that is constructed via the fused matmul of 
    two tensors, tensor A that is generated via prng function and tensor B, a trainable
    coefficient tensor. Tensor A has shape (n, int_dim). Tensor B has shape (int_dim,).
    The output tensor will be reshaped into an arbitrary shape where n is the number of elements.
    
    This custom tensor subclass should be usable anywhere, however when utilising it inside nn.Module, it is
    necessary to manually register the _coef tensor as a parameter of the module for any optimizers and stuff to work:
    `self.register_parameter("coef", nn.Parameter(self.rt._coef))`
    
    Do note that due to the nature of this tensor, device is expected to be 'cuda' upon initialization and basically all the time.
    Set requires_grad=True at the start, do not reassign it post-init as this might break the subclass.
    
    Be aware there is a bug if you use device="cuda" in the __init__ pytorch will throw an error:
    `RuntimeError: 0 <= device.index() && device.index() < static_cast<c10::DeviceIndex>(device_ready_queues_.size()) INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/engine.cpp":1522, please report a bug to PyTorch.`
    when it tries to call the backward pass. I have no clue why this happens, but it is likely related to the .to(..., *args, **kwargs) where one of the extra arguments is causing the coef tensor to map to an invalid device?
    To prevent this, just init the tensor on CPU then use .to("cuda")
    """
    
    @staticmethod
    def __new__(cls, data: torch.Tensor, seed: int, shape: Union[tuple, torch.Size], requires_grad=False):
        
        # Creating the wrapper will generally detach tensors from the autograd graph, ensure that the 
        # input does not require grad (should be taken care of by constructor Function)
        assert not data.requires_grad or not torch.is_grad_enabled()
        
        return torch.Tensor._make_wrapper_subclass(cls, torch.Size(shape), requires_grad=False)

    def __init__(self, data: torch.Tensor, seed: int, shape: Union[tuple, torch.Size], requires_grad=False):
        """
        Args:
            data (torch.Tensor): If `int`, represents intermediate/intrinsic dimension, and random coef
                tensor will be generated. If `Iterable`, will use that coefs as the coef tensor.
            seed (int): Seed for the prng generator
            shape (Union[tuple, torch.Size]): Final tensor output shape
            dtype (torch.dtype): dtype
            device (str, optional): device variable cannot actually be properly updated with _C.TensorBase class. 
                This is instead only used for initialising _coef device, the class's device type will always follow _coef.
                Defaults to 'cpu'.
            requires_grad (bool, optional): Requires grad. Defaults to False.
        """
        self.seed = seed        
        
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
            return f"RandumbTensor(seed={self.seed}, int_dim={self.int_dim}, shape={self.shape}, noise_matrix_shape={self.noise_matrix_shape}, dtype={self.dtype}, device='{self._coef.device}', coef={self._coef})"
    
    def get_materialized(self):
        # helper function to materialize the full matrix, should only be used for debugging tbh, for all intents and purposes other than
        # printing the output tensor, just calling rt should suffice as the torch_dispatch override should materialize the matrix anyway
        self.materialize()
        ref = self._tensor
        self.dematerialize()
        return ref
    
    #TODO: the distribution of the noise matrix and therefore final output tensor seems to be incredibly biased?
    def get_noise_matrix(self):
        # helper function to materialize the standalone noise matrix, should only be used for debugging
        true_noise_mat = torch.zeros(self.noise_matrix_shape, device=self._coef.device)
        for row in range(self.noise_matrix_shape[0]):
            row_vals = squirrel5_generate(row, size=self.noise_matrix_shape[1], BLOCK_SIZE=256, seed=self.seed)
            true_noise_mat[row] = row_vals
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
        
    def materialize(self):
        # This should be outside the autograd computation graph already if ever called, but just in case we wrap it in torch.no_grad
        with torch.no_grad():
            # materializes the tensor, then reshape it appropriately. 
            self._tensor = materialize_fwd(self._coef, self.noise_matrix_shape[0], self.seed).view(self.shape)
    
    def dematerialize(self):
        # this function should free _tensor from memory. We can safely delete the entire tensor because the autograd computation is tracked manually by the
        # wrapper creation Function.
        del self._tensor
        return
    
    
    __torch_function__ = torch._C._disabled_torch_function_impl
    #TODO: support .detach() returning a still wrapped RT, due to https://github.com/pytorch/pytorch/issues/77265#issuecomment-1129044684
    # its supposed to be overridden in torch_dispatch, but we can check if its possible to override it in torch_function instead.
    # this is so that it will work with nn.Parameter without pytorch shitting itself.
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
        # print(f"dispatched {func} {args} {kwargs}")
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
    def forward(ctx, coefs: torch.Tensor, seed: int, shape: Iterable) -> RandumbTensor:
        if type(coefs) is torch.Tensor or torch.nn.Parameter:
            assert len(coefs.size()) == 1, f"Coefficient tensor must be 1D but received tensor of shape {coefs.size()}"
            out = RandumbTensor(coefs, seed, shape)

            ctx.dcoefs_size = coefs.numel()
            ctx.seed = seed
            return out
        else:
            raise AssertionError(f"Expected coefs to be of type torch.Tensor, found {type(coefs)} instead.")

    @staticmethod
    def backward(ctx, grad):
        # print(f"backward pass called with grad {grad}")
        # flatten the grad back to same shape as materialize bwd output
        grad = grad.view(-1)
        # run the actual bwd
        dout = materialize_bwd(grad, ctx.dcoefs_size, ctx.seed)
        return dout, None, None
    
    @classmethod
    def apply(cls, coefs: torch.Tensor, seed: int, shape: Iterable) -> RandumbTensor:
        # trivial override of the .apply signature that's actually `apply(cls, *args, **kwargs)` just to make typehinting work because it's annoying me
        return super().apply(coefs, seed, shape)

# alias for the constructor, use this as `rt = CreateRandumbTensor(...)`
CreateRandumbTensor = RandumbTensorConstructor.apply


if __name__ == "__main__":
    
    def getBack(var_grad_fn):
        print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print(n[0])
                    print('Tensor with grad found:', tensor)
                    print(' - gradient:', tensor.grad)
                    print()
                except AttributeError as e:
                    getBack(n[0])
                
    torch.manual_seed(0)
    DEVICE = "cuda" 
    seed = 1337
    output_shape = (5, 10)
    w1 = nn.Parameter(torch.randn(7, requires_grad=True, device=DEVICE))
    w2 = torch.randn((10, 3), requires_grad=False, device=DEVICE)

    # Autograd doesn't "unwrap" variables, they still remember if they
    # requires_grad; and in fact, inside __torch_dispatch__ it is willing
    # to mix gradients between multiple levels. The current class does
    # catch most of these though when it is looking at the different
    # arguments
    x = CreateRandumbTensor(w1, seed, output_shape)
    print(f"x {x}")
    print("performing backward")
    # there are two different behaviours here. if the w2 is of a normal tensor, the grads are not backpropagated
    # to the inner tensors of the tensor subclass.
    # but if w2 is of InnerAutogradTensor then the gradients are properly backpropagated?
    for i in range(10):
        print(f"innerautograd backward {i}")
        y = x @ w2
        y.sum().backward(retain_graph=True)
    print(w1.grad)
    print("x inner grads", x._coef.grad)
    
    print("--------------")
    print("getting backwards graph of y.gradfn")
    getBack(y.grad_fn)
    
    print("--------real values---------")
    true_coef = w1.detach().clone().to(DEVICE)
    true_coef.requires_grad_()
    n, m = torch.Size(output_shape).numel(), true_coef.size(0)
    true_noise_mat = torch.zeros((n, m), device=DEVICE, requires_grad=False)
    for row in range(n):
        row_vals = squirrel5_generate(row, size=m, BLOCK_SIZE=256, seed=seed)
        true_noise_mat[row] = row_vals

    true_rt = (true_noise_mat @ true_coef).view(output_shape)
    print(f"checking truert equal rt {torch.allclose(x, true_rt)}")
    true_w2 = w2.detach().clone()
    for i in range(10):
        print(f"true backward {i}")
        y = true_rt @ true_w2
        y.sum().backward(retain_graph=True)
    
    print(f"true coef grad {true_coef.grad}")
    