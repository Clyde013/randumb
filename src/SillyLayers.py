# implements silly little custom nn.Modules for common pytorch layers, except the weights
# are composed of RandumbTensors as their parameters.
import math
import warnings

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import _calculate_correct_fan, calculate_gain
from torch import Generator

from torch.nn.common_types import _size_2_t

from src.RandumbTensor import CreateRandumbTensor, RandumbTensor



def init_kaiming_uniform(weight: RandumbTensor, bias: RandumbTensor, seed: int, a: float = math.sqrt(5), mode: str = "fan_in", nonlinearity: str = "relu", generator: Generator | None = None) -> None:
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
        nonlinearity (str, optional): Default pytorch is leaky_relu, but since we use relu layers. Defaults to "relu".
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


class SillyLinear(nn.Module):
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

    __constants__ = ["in_features", "out_features", "int_dim", "seed"]
    in_features: int
    out_features: int
    int_dim: int
    seed: int
    weight: RandumbTensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        int_dim: int,
        seed: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.int_dim = int_dim
        self.seed = seed
        
        self.weight_coef = nn.Parameter(torch.empty(int_dim, dtype=dtype, device=device), requires_grad=True)
        self.weight = CreateRandumbTensor(self.weight_coef, seed, (out_features, in_features))
        
        if bias:
            self.bias_coef = nn.Parameter(torch.empty(int_dim, dtype=dtype, device=device), requires_grad=True)
            self.bias = CreateRandumbTensor(self.bias_coef, seed+1, (out_features, ))
        else:
            self.bias = None
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_kaiming_uniform(self.weight, self.bias, self.seed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int_dim={self.int_dim}, seed={self.seed}, bias={self.bias is not None}"



class SillyConv2d(nn.Conv2d):
    """
    Adapted from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Basically a conv2d layer.
    """
    weight: RandumbTensor
    bias: RandumbTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_dim: int,
        seed: int,
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
        self.int_dim = int_dim
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
        self.weight_coef = nn.Parameter(torch.empty(self.int_dim, dtype=self.weight.dtype, device=self.weight.device), requires_grad=True)
        weight_shape = self.weight.shape
        del self.weight
        self.weight = CreateRandumbTensor(coefs=self.weight_coef, seed=self.seed, shape=weight_shape)
        
        if self.bias is not None:
            self.bias_coef = nn.Parameter(torch.empty(self.int_dim, dtype=self.bias.dtype, device=self.bias.device), requires_grad=True)
            bias_shape = self.bias.shape
            del self.bias
            self.bias = CreateRandumbTensor(coefs=self.bias_coef, seed=self.seed+1, shape=bias_shape)
        
        init_kaiming_uniform(self.weight, self.bias, self.seed)

    def extra_repr(self) -> str:
        return f"int_dim={self.int_dim}, seed={self.seed}"



if __name__ == "__main__":
    """
    fc1 = SillyLinear(9216, 128, 64, 10)
    fc1.to("cuda")
    for name, param in fc1.named_parameters():
        if param.requires_grad:
            print(name)
    """
            