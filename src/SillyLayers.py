# implements silly little custom nn.Modules for common pytorch layers, except the weights
# are composed of RandumbTensors as their parameters.
import math
import warnings

from typing import Union, Optional, Generator
from types import GeneratorType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import _calculate_correct_fan, calculate_gain
from torch import Generator

from torch.nn.common_types import _size_2_t

from src.RandumbTensor import CreateRandumbTensor, RandumbTensor, embed_fwd



# we implement two trivial generators for model seeding and int_dim assignment. 
# stepped generator stays true to the idea of compressing the model as much as possible, and allows
# the entire model to be stored as a single offset and stepsize.
def stepgen(offset, step):
    i = 0
    while True:
        yield offset + step * i
        i += 1
# list generator allows the seeds of every RT to be stored sequentially,
# in case you somehow "fine-tune" seeds (which sounds silly enough to be implemented in the future...)
def listgen(lst):
    for i in lst:
        yield i


def init_kaiming_uniform(weight: RandumbTensor, bias: Optional[RandumbTensor], seed: int, a: float = math.sqrt(5), mode: str = "fan_in", nonlinearity: str = "leaky_relu", generator: Generator | None = None) -> None:
    """
    This function is basically the entirity of any nn.Module layer's reset_parameters() function. Since this sequence of weight initialisation is shared by both linear and conv layers,
    the code is put into this function instead of duplicating the reset_parameters() for both the custom Linear and Conv layers.
    See: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py#L114

    Args:
        weight (RandumbTensor): self.weight of the layer
        bias (RandumbTensor): self.bias of the layer (can be None)
        seed (int): self.seed of the layer
        a (float, optional): slope of the leaky_relu, shouldn't matter. Defaults to math.sqrt(5).
        mode (str, optional): should be always fan_in since we want to normalize values of the forward pass. Defaults to "fan_in".
        nonlinearity (str, optional): Default pytorch is leaky_relu. Ref: https://github.com/pytorch/pytorch/issues/15314 Defaults to "linear".
        generator (Generator | None, optional): Should be a generator created with seed, initialized below. Defaults to None.
    """

    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    if generator is None: generator = torch.Generator(device=weight._coef.device).manual_seed(seed)
    if 0 in weight._coef.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return weight._coef

    # this snippet is just `kaiming_uniform_`` https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L456
    fan_in = _calculate_correct_fan(weight, mode)
    gain = calculate_gain(nonlinearity, a)
    # this is the "true" intended distribution of the final materialized matrix
    out_std = gain / math.sqrt(fan_in)
    
    # $\text{var}(y)=\frac{3}{d_y}std^2$ where $d_y$ is the coef dim.
    coef_std = math.sqrt(3/weight.int_dim)*out_std
    bound = math.sqrt(3.0) * coef_std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        weight._coef.uniform_(-bound, bound, generator=generator)
    
    # similar for bias
    if bias is not None:
        # as per nn.Linear implementation, we utilize fan_in and std computed from self.weight when initializing bias
        bias_std = math.sqrt(3/bias.int_dim)*out_std
        bound = math.sqrt(3.0) * bias_std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            bias._coef.uniform_(-bound, bound, generator=generator)


class SillyModuleMixin(nn.Module):
    """
    This mixin is used to implement certain shared functionality that we need to override from nn.Module.
    """
    
    def _apply(self, fn):
        # General practice when instantiating models is creating tensors on cpu then moving the whole model using .to(), however if coef is instantiated on 'cpu' and later moved onto the gpu:
        # 1. RTs are not automatically updated (can be fixed using 'register_buffer') 
        # 2. PyTorch silently refuses to change the device, and explicitly doing so causes `AttributeError: attribute 'device' of 'torch._C.TensorBase' objects is not writable`
        # the 2nd one is the main problem, since once RT is created using _make_wrapper_subclass, we can never update the 'device' again without creating a new obj. Pytorch will error since
        # the rt.device mismatches - even if in theory all the code would run since there's no actual tensor being hosted by the subclass on the CPU.
        # This override of ._apply() will simply recreate a new RandumbTensor with the correct device/dtype whenever Module functions such as .to(), .cuda() or .float()/.bfloat16() are called.
        def wrapped_fn(inp):
            if isinstance(inp, RandumbTensor):
                # CreateRandumbTensor will create RT of same dtype and device as the _coef. Per https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/module.py#L923 the 
                # parameters are updated first whenever ._apply() is called, so recreating buffers will always have the rt._coef of correct dtype and device without us having to explicitly
                # pass them in. Note that since we are creating an entirely new RT instead of dispatching to aten._to_copy, the view_ops has to be manually copied
                # in case there were any view_ops applied onto the tensor before the ._apply(). While this should not be the case, it's better to be safe.
                new_rt = CreateRandumbTensor(inp._coef, inp.seed, inp.shape)
                new_rt.view_ops = inp.view_ops
                new_rt._init_shape = inp._init_shape
                return new_rt
            else:
                return fn(inp)
        super()._apply(wrapped_fn)
        return self


class SillyLinear(SillyModuleMixin, nn.Module):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
    Weight and bias are replaced with RandumbTensors.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        int_dim: intermediate/intrinsic dimension used for RandumbTensors
        seed: seed of the RandumbTensor
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from RandumbTensors.
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from RandumbTensors.
    """

    __constants__ = ["in_features", "out_features", "int_dim"]
    in_features: int
    out_features: int
    int_dim: int
    seed: Union[int, list]
    weight: RandumbTensor
    # number of seeds expected for set_seed()
    num_seeds = 2

    def __init__(
        self,
        in_features: int,
        out_features: int,
        int_dim: Union[list[int], Generator],
        seed: Union[list[int], Generator],
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # accept both generator and just the list of int_dim/seeds
        if isinstance(int_dim, GeneratorType):
            self.int_dim = [next(int_dim) for i in range(self.num_seeds)]
        else: 
            self.int_dim = int_dim
        if isinstance(seed, GeneratorType):
            self.seed = [next(seed) for i in range(self.num_seeds)]
        else:
            self.seed = seed
        
        self.weight_coef = nn.Parameter(torch.empty(self.int_dim[0], dtype=dtype, device=device), requires_grad=True)
        # self.weight = CreateRandumbTensor(self.weight_coef, self.seed[0], (out_features, in_features))
        self.register_buffer("weight", CreateRandumbTensor(self.weight_coef, self.seed[0], (out_features, in_features)))
        
        if bias:
            self.bias_coef = nn.Parameter(torch.empty(self.int_dim[1], dtype=dtype, device=device), requires_grad=True)
            # self.bias = CreateRandumbTensor(self.bias_coef, self.seed[1], (out_features, ))
            self.register_buffer("bias", CreateRandumbTensor(self.bias_coef, self.seed[1], (out_features, )))
        else:
            self.bias = None
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_kaiming_uniform(self.weight, self.bias, self.seed[0])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int_dim={self.int_dim}, seeds={self.seed}, bias={self.bias is not None}"


class SillyConv2d(SillyModuleMixin, nn.Conv2d):
    """
    Adapted from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Basically a conv2d layer.
    """
    weight: RandumbTensor
    bias: RandumbTensor
    # number of seeds/int_dim expected
    num_seeds = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_dim: Union[list[int], Generator],
        seed: Union[list[int], Generator],
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # accept both generator and just the list of int_dim/seeds
        if isinstance(int_dim, GeneratorType):
            self.int_dim = [next(int_dim) for i in range(self.num_seeds)]
        else: 
            self.int_dim = int_dim
        if isinstance(seed, GeneratorType):
            self.seed = [next(seed) for i in range(self.num_seeds)]
        else:
            self.seed = seed

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def reset_parameters(self) -> None:
        # the super().__init__() of the conv instantiates their own self.weight, we just copy the shape but replace it with our own tensor subclass
        self.weight_coef = nn.Parameter(torch.empty(self.int_dim[0], dtype=self.weight.dtype, device=self.weight.device), requires_grad=True)
        weight_shape = self.weight.shape
        del self.weight
        self.register_buffer("weight", CreateRandumbTensor(coefs=self.weight_coef, seed=self.seed[0], shape=weight_shape))
        
        if self.bias is not None:
            self.bias_coef = nn.Parameter(torch.empty(self.int_dim[1], dtype=self.bias.dtype, device=self.bias.device), requires_grad=True)
            bias_shape = self.bias.shape
            del self.bias
            self.register_buffer("bias", CreateRandumbTensor(coefs=self.bias_coef, seed=self.seed[1], shape=bias_shape))
        
        init_kaiming_uniform(self.weight, self.bias, self.seed[0])
        # init_zeros(self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"int_dim={self.int_dim}, seeds={self.seed}"


# TODO: can probably create a custom kernel that fast lookups the indices, instead of materializing the whole thing and relying
# on the builtin aten embedding fwd and bwd (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp)
class SillyEmbedding(SillyModuleMixin, nn.Embedding):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/sparse.py#L15
    Does not support padding_idx.
    """
    # number of seeds/int_dim expected
    num_seeds = 1
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        int_dim: Union[list[int], Generator],
        seed: Union[list[int], Generator],
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[RandumbTensor] = None,
        _freeze: bool = False,
        device="cuda",  # device should always be cuda since it relies on RT triton kernels
        dtype=None,
    ) -> None:
        # supercall the parent of nn.Embedding which is nn.Module
        super(nn.Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # accept both generator and just the list of int_dim/seeds
        if isinstance(int_dim, GeneratorType):
            self.int_dim = [next(int_dim) for i in range(self.num_seeds)]
        else: 
            self.int_dim = int_dim
        if isinstance(seed, GeneratorType):
            self.seed = [next(seed) for i in range(self.num_seeds)]
        else:
            self.seed = seed
            
        assert padding_idx is None, "padding_idx not supported"
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight_coef = nn.Parameter(torch.empty(self.int_dim[0], dtype=dtype, device=device), requires_grad=not _freeze)
            self.register_buffer("weight", CreateRandumbTensor(self.weight_coef, self.seed[0], (num_embeddings, embedding_dim)))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], f"Shape of weight {_weight.shape} does not match num_embeddings and embedding_dim"
            self.register_buffer("weight", _weight)
            self.weight_coef = self.weight._coef

        self.sparse = sparse

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return embed_fwd(input, self.weight_coef, self.embedding_dim, self.weight.seed)

    def reset_parameters(self) -> None:
        init_kaiming_uniform(self.weight, None, seed=self.seed[0])
        # self._fill_padding_idx_with_zero()
        

if __name__ == "__main__":
    """
    fc1 = SillyLinear(9216, 128, 64, 10)
    fc1.to("cuda")
    for name, param in fc1.named_parameters():
        if param.requires_grad:
            print(name)
    """
            