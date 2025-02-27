# implements silly little custom nn.Modules for common pytorch layers, except the weights
# are composed of RandumbTensors as their parameters.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import _calculate_correct_fan, calculate_gain
from torch import Generator

from src.RandumbTensor import CreateRandumbTensor, RandumbTensor


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
        
        self.weight_coef = nn.Parameter(torch.randn(int_dim, dtype=dtype, device=device), requires_grad=True)
        self.weight = CreateRandumbTensor(self.weight_coef, seed, (out_features, in_features))
        
        if bias:
            self.bias_coef = nn.Parameter(torch.randn(int_dim, dtype=dtype, device=device), requires_grad=True)
            self.bias = CreateRandumbTensor(self.bias_coef, seed+1, (out_features, ))
        else:
            self.bias = None
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        tensor = self.weight._coef
        a = math.sqrt(5)
        nonlinearity: str = "relu"
        generator = torch.Generator(device=tensor.device).manual_seed(self.seed)
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor

        # this snippet is just `kaiming_uniform_`` https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L456
        fan_in = _calculate_correct_fan(self.weight, "fan_in")
        gain = calculate_gain(nonlinearity, a)
        # this is the "true" intended distribution of the final materialized matrix
        out_std = gain / math.sqrt(fan_in)
        
        # $\text{var}(y)=\frac{3}{d_y}std^2$ where $d_y$ is the coef dim.
        coef_std = math.sqrt(3/self.weight.int_dim)*out_std
        bound = math.sqrt(3.0) * coef_std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            tensor.uniform_(-bound, bound, generator=generator)
            
        # similar for bias
        if self.bias is not None:
            # as per nn.Linear implementation, we utilize fan_in and std computed from self.weight when initializing bias
            bias_std = math.sqrt(3/self.bias.int_dim)*out_std
            bound = math.sqrt(3.0) * bias_std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                self.bias._coef.uniform_(-bound, bound, generator=generator)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int_dim={self.int_dim}, seed={self.seed}, bias={self.bias is not None}"


if __name__ == "__main__":
    """
    fc1 = SillyLinear(9216, 128, 64, 10)
    fc1.to("cuda")
    for name, param in fc1.named_parameters():
        if param.requires_grad:
            print(name)
    """
            