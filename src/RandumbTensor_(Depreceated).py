from typing import Union
from collections.abc import Iterable

import torch
from torch.utils._pytree import tree_map, tree_map_only
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton, custom_op

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
    # print("backward pass called")
    dcoefs = materialize_bwd(grad, ctx.dcoefs_size, ctx.seed)
    # only coefs needs grad, the rest are non-differentiable. function signature of fwd: coefs, size, seed, device
    return dcoefs, None, None, None

def setup_context(ctx, inputs, output):
    # print("setup context")
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
    def __new__(cls, data: Union[Iterable, int], seed: int, shape: Union[tuple, torch.Size], dtype: torch.dtype = torch.float32, device='cpu', requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(cls, torch.Size(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    def __init__(self, data: Union[Iterable, int], seed: int, shape: Union[tuple, torch.Size], dtype: torch.dtype = torch.float32, device='cpu', requires_grad=False):
        """
        Args:
            data (Union[Iterable, int]): If `int`, represents intermediate/intrinsic dimension, and random coef
                tensor will be generated. If `Iterable`, will use that input as the coef tensor.
            seed (int): Seed for the prng generator
            shape (Union[tuple, torch.Size]): Final tensor output shape
            dtype (torch.dtype): dtype
            device (str, optional): device variable cannot actually be properly updated with _C.TensorBase class. 
                This is instead only used for initialising _coef device, the class's device type will always follow _coef.
                Defaults to 'cpu'.
            requires_grad (bool, optional): Requires grad. Defaults to False.
        """
        self.seed = seed
        self._requires_grad = requires_grad
        
        # this is a hacky manual way to denote the tensor as a nn.Parameter that should be registered by nn.Module. 
        # if this is not here, the .to() and other methods applied on nn.Module will not be transmitted to this tensor.
        self._is_param = True
        
        if isinstance(data, int):
            # initialise empty coefficient tensor if only size is provided
            self._coef = torch.randn(data, dtype=self.dtype, requires_grad=requires_grad, device=device)
            self.int_dim = data
        else:
            # initialise coefficient tensor from given data
            self._coef = torch.as_tensor(data, dtype=self.dtype, device=device)
            if self._requires_grad: self._coef.requires_grad_()
            size = self._coef.size()
            assert len(size) == 1, f"Coefficient tensor must be 1D but received tensor of shape {self.size}"
            self.int_dim = size[0]
        
        # final output tensor will be materialised in this reference whenever necessary
        self._tensor: torch.Tensor = None
        # shape of the prng generated tensor (that should never be fully materialised)
        self.noise_matrix_shape = torch.Size((self.shape.numel(), self.int_dim))
        
        # register the custom post accumulation grad hook (right now it assumes requires_grad is true upon init all the time)
        if self._requires_grad:
            self._coef_grad_hook = self.register_post_accumulate_grad_hook(self._coef_grad_hook_func)
    
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
        with record_function("materialize"):
        # materializes the tensor, then reshape it appropriately
            self._tensor = materialize_fwd(self._coef, self.noise_matrix_shape[0], self.seed).reshape(self.shape)
        # print("tensor mat")
    
    def dematerialize(self):
        # this function should free _tensor's underlying storage from memory. We don't delete the tensor because we still need the tensor's metadata on the
        # autograd graph in order to properly backward pass. So we instead manipulate the underlying tensor memory storage by setting it to an empty one,
        # and then later rematerialize the storage.
        with record_function("dematerialize"):
            empty_storage = torch.empty((0,), dtype=self._tensor.dtype, device=self._tensor.device)
            self._tensor.set_(empty_storage)
        # self._tensor_gradfn = self._tensor.grad_fn
        # print("tensor demat")
        # del self._tensor
        return
    
    def rematerialize(self):
        with record_function("rematerialize"):
        # rematerializes the tensor's underlying storage for the backward pass in the same way as .materialize(), but without grad tape associated
            with torch.no_grad():
                self._tensor.set_( materialize_fwd(self._coef, self.noise_matrix_shape[0], self.seed).reshape(self.shape) )
        
        # with torch.no_grad():
        #     self._tensor = materialize_fwd(self._coef, self.noise_matrix_shape[0], self.seed).reshape(self.shape)
        # self._tensor.grad_fn = self._tensor_gradfn
        # self._tensor = materialize_fwd(self._coef, self.noise_matrix_shape[0], self.seed).reshape(self.shape)
        # print("tensor remat")
        return
    
    @staticmethod
    def _coef_grad_hook_func(self):
        """
        Realistically this shouldn't be a class function, just because of how the function signature for hooks are supposed
        to be defined (hook(param: Tensor) -> None), but I think it's neater to have a private hook for this class be 
        self-contained within the class definition, so I sort of abuse @staticmethod with calling from self in the init.
        So the "self" here is actually just the param and not implicit first argument self instance, although both are equivalent.
        
        This post accumulation grad hook should trigger once (obviously) the grads for the outer wrapper tensor have been
        computed. Then we can call our custom backward function utilizing the grads for the outer tensor to compute the
        backward pass and gradients for the _coef.
        
        If you are wondering why we implement the backward using a hook instead of integrating it into the autograd in some more elegant way,
        the answer is that a more elegant way does not exist. See: https://github.com/pytorch/pytorch/issues/108983#issuecomment-1714706014
        It seems the two autograd levels will remain separate no matter what, any attempts to combine them will probably be a massive headache and 
        end up causing big issues. The custom function fwd bwd pass wrapping the instantiation (i.e. negativetensor https://github.com/albanD/subclass_zoo/blob/main/negative_tensor.py)
        will not work, due to bug with ctx.save_for_backward (https://github.com/pytorch/pytorch/issues/47117#issuecomment-901138682), otherwise we can
        possibly have the _coef tensor be instantiated separately outside the tensor and then compute the bwd pass manually to backprop gradients to it
        through the custom function bwd, but unfortunately this requires having a reference to the custom class in the bwd, which is not possible due to bug.
        Even if we could, we would have to write the backward pass ourselves anyway, as the computation graph for the inner tensors won't be composed either...
        Now, the bset band-aid solution implemented here is to have the two autograd computation graphs remain separate, and call the hook once this "leaf" node gets
        the grads accumulated, and then call the self._tensor.backward() which has the separate autograd graph, using the self.grad accumulated into the
        outer wrapper tensor. This means we have to manually ensure that autograd computes the gradients correctly with edge cases such as grad accumulation,
        as well as when freeing the self._tensor from memory we have to somehow maintain the autograd computation graph, probably by freeing the self._tensor
        via https://pytorch.org/docs/stable/storage.html
        """
        with record_function("bwd grad hook"):
            if self._coef.requires_grad:
                # following torch optimizer.zero_grad() implementation to zero the _coef grads manually: https://github.com/pytorch/pytorch/blob/46e83bb6377ad11c475fafc93c9ea15433056573/torch/optim/optimizer.py#L965
                # this should ensure that gradient accumulation works as intended, the gradients will be accumulated into the wrapper tensor, and the correct gradient for the _coef should be a single backward wrt. the
                # accumulated wrapper gradient, not accumulating the accumulated gradient (if that makes sense).
                if self._coef.grad is not None:
                    self._coef.grad.detach_()
                    self._coef.grad.zero_()
                #TODO: figure out a good way to rematerialize the _tensor storage?
                self.rematerialize()
                self._tensor.backward(self.grad)
    
    #TODO: support .detach() returning a still wrapped RT, due to https://github.com/pytorch/pytorch/issues/77265#issuecomment-1129044684
    # its supposed to be overridden in torch_dispatch, but we can check if its possible to override it in torch_function instead.
    # this is so that it will work with nn.Parameter without pytorch shitting itself.
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # keep track of all materialized tensors
        materialized_tensors: list[RandumbTensor] = []
        def _materialize(e):
            if isinstance(e, RandumbTensor):
                # by default pytorch prevents recursive dispatching to avoid infinite loops, but this also prevents our materialize_fwd
                # and its respective backwards pass from being properly tracked in the computation graph (i.e. _tensor has no graph_fn)
                # to fix this we have to wrap it in enable_reentrant_dispatch()
                # note that the autograd computation graph created by this is considered a "separate level" of autograd, and will not
                # compose with the autograd graph of the outer wrapper tensor (.backward() will not cause grads to accumulate into _tensor's leafs)
                # see above hook for current solution where we manually call .backward() on the inner autograd graph ourselves.
                with torch.overrides.enable_reentrant_dispatch():
                    e.materialize()
                    materialized_tensors.append(e)
                    return e._tensor.view_as(e._tensor)
            else: 
                return e
            
        # perform the operation with materialized tensors
        # print(f"dispatched {func} {args} {kwargs}")
        ret = func(*tree_map(_materialize, args), **tree_map(_materialize, kwargs))
        
        # dematerialize all previously materialized tensors
        for t in materialized_tensors:
            t.dematerialize()
            
        return ret
    
    
    # overrides default torch.Tensor.to behaviour, as we want to update the _coef device
    # as that is the only "real" tensor that this object has.
    def _to(super_fn, self, device, *args, **kwargs):
        if isinstance(self, RandumbTensor):
            self._coef = self._coef.to(device, *args, **kwargs)
            # take note to set the requires_grad again as after the .to() op it is considered a new tensor and not a leaf
            # ref: https://discuss.pytorch.org/t/none-grad-attribute-in-multiprocessing-autograd-with-floattensor-type/20482/2
            if self._requires_grad: self._coef.requires_grad_()
            return self
        else:
            return super_fn(self, device, *args, **kwargs)
    
    IMPLEMENTATIONS = {torch.Tensor.to: _to}
    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        def super_fn(*args, **kwargs):
            # Disable torch_function by hand because we don't want the wrapping behavior of
            # the super() impl. Default torch_function will always wrap outputs into a subclass if they aren't already a subclass.
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)
                """
                try:
                    print(f"{func} {args} {kwargs}")
                    return func(*args, **kwargs)
                except:
                    return func(*args, **kwargs)
                """

        if func in self.IMPLEMENTATIONS:
            # note: functions that are intercepted don't trigger __torch_dispatch__
            try:
                # print(f"{func} {args} {kwargs}")
                return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})
            except Exception as e:
                print(e)
                raise e
        # This is just a no-op for all the non-factory functions:
        return super_fn(*args, **kwargs or {})
    
    # TODO: implement indexing and slicing like https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/uint4_tensor.py#L89
    
    
