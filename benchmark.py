import matplotlib.pyplot as plt
import numpy as np
import math

import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
from torch.utils._pytree import tree_map, tree_map_only
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

import triton
import triton.language as tl

from rten_cpp.RandumbTensor import CreateRandumbTensor, squirrel5_generate_2d, materialize_fwd, materialize_bwd, rt_mm
from rten_cpp.SillyLayers import SillyLinear

from rten_cpp.utils import _test_memory

DEVICE = "cuda"
@torch.compile
def torch_rand_compiled(coeffs, size, m):
    noise_mat = torch.randn((size, m), device=DEVICE)
    out = noise_mat @ coeffs
    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 23, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        y_log=True,
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch_rand', 'torch_rand_compiled', 'materialize'],  # Possible values for `line_arg`.
        line_names=['torch_rand', 'torch_rand_compiled', 'materialize'],  # Label name for the lines.
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='custom prng speed performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_speed_materialize(size, provider):
    """
    Benchmarks the materialization speed of the tensor vs a naive torch implementation and simple torch.compile one.
    """
    quantiles = [0.5, 0.2, 0.8]
    m = round(size * 0.01)
    coefs = torch.randn(m, device=DEVICE, dtype=torch.float32)
    scale_bias = torch.tensor([1], device=DEVICE, dtype=torch.float32)
    
    def materialize_test():
        return materialize_fwd(coefs, size, 1337, scale_bias, True)
    
    def torch_rand():
        noise_mat = torch.randn((size, m), device=DEVICE)
        out = noise_mat @ coefs
        return out

    def torch_rand_compiled_wrapper():
        return torch_rand_compiled(coefs, size, m)
    
    if provider == 'torch_rand':
        if size > 2**19:
            ms, min_ms, max_ms = None, None, None
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(torch_rand, quantiles=quantiles)
        
    if provider == 'torch_rand_compiled':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_rand_compiled_wrapper, quantiles=quantiles)
        
    if provider == "materialize":
        ms, min_ms, max_ms = triton.testing.do_bench(materialize_test, quantiles=quantiles)
        
    return ms, max_ms, min_ms
    
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 20, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch_rand', 'torch_rand_compiled', 'materialize'],  # Possible values for `line_arg`.
        line_names=['torch_rand', 'torch_rand_compiled', 'materialize'],  # Label name for the lines.
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],  # Line styles.
        ylabel='MiB',  # Label name for the y-axis.
        plot_name='custom prng peak memory usage',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_mem_materialize(size, provider):
    quantiles = [0.5, 0.2, 0.8]
    m = 2048
    coefs = torch.randn(m, device=DEVICE, dtype=torch.float32)
    scale_bias = torch.tensor([1], device=DEVICE, dtype=torch.float32)
    
    def materialize_test():
        return materialize_fwd(coefs, size, 1337, scale_bias, True)
    
    def torch_rand():
        noise_mat = torch.randn((size, m), device=DEVICE)
        out = noise_mat @ coefs
        return out

    def torch_rand_compiled_wrapper():
        return torch_rand_compiled(coefs, size, m)
    
    if provider == 'torch_rand':
        mem_50, mem_20, mem_80 = _test_memory(
            torch_rand,
            quantiles=quantiles,
        )
        
    if provider == 'torch_rand_compiled':
        mem_50, mem_20, mem_80 = _test_memory(
            torch_rand_compiled_wrapper,
            quantiles=quantiles,
        )
        
    if provider == "materialize":
        mem_50, mem_20, mem_80 = _test_memory(
            materialize_test,
            quantiles=quantiles,
        )
        
    return mem_20, mem_50, mem_80
    

def check_func():
    torch.set_printoptions(edgeitems=20)
    n, m = 18, 18
    coefs = torch.randn(m, device=DEVICE, dtype=torch.float32)
    true_coefs = coefs.detach().clone()
    true_coefs.requires_grad = True
    t = materialize_fwd(coefs, n, seed=1337, scale_bias=None, fused_bias=False)
    print("materialized matrix:")
    print(t)
    y = t.cpu().detach().numpy()
    
    # check the output is actually matching what it is expected to be
    true_noise_mat = torch.zeros((n, m), device=DEVICE)
    for row in range(n):
        row_vals = squirrel5_generate_2d(row, size=m, BLOCK_SIZE=256, seed=1337)
        true_noise_mat[row] = row_vals
    true_out = true_noise_mat @ true_coefs    
    print(f"dist between calculated and true: {torch.dist(true_out, t)}")

    # now that we've computed the output, we need to create an overly elaborate function to check the backward pass
    true_out = true_out.reshape(3, 6)
    
    rt_coef = coefs.detach().clone()
    rt_coef.requires_grad_()
    rt = CreateRandumbTensor(rt_coef, 1337, (3, 6))

    print(f"dist between rt and true: {torch.dist(rt, true_out)}")
    
    
    x = torch.randn(6, 12, device=DEVICE, requires_grad=False)
    linear_layer = nn.Linear(12, 8, device=DEVICE)
    
    
    rt_eqn = linear_layer(rt @ x).sum()
    print(f"rt_eqn out: {linear_layer(rt @ x)}")
    
    
    true_eqn = linear_layer(true_out @ x).sum()
    print(f"true_eqn out: {linear_layer(true_out @ x)}")
    
    # now we check the gradient propagations
    true_eqn.backward()
    rt_eqn.backward()
    
    print(f"dist between rt and true: {torch.dist(rt, true_out)}")
    print(f"grad for rt coef: {rt._coef.grad}")
    print(f"grad for true: {true_coefs.grad}")
    print(f"grad diff: {torch.dist(rt._coef.grad, true_coefs.grad)}")

    # quick and dirty visual check to see if squirrel5 value distribution is statistically acceptable
    # seems to be normally distributed in histogram (within reason) and passes the Kolmogorov–Smirnov test
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.hist(y, bins='auto')
    ax2.scatter(y, range(y.size))
    ax3.plot(np.sort(y))
    plt.savefig("distribution.png") 
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(6, 20, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'rt', 'rt_fused'],  # Possible values for `line_arg`.
        line_names=['torch', 'rt', 'rt_fused'],  # Label name for the lines.
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='torch vs RT speed performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_speed_rt(size, provider):
    """
    Benchmarks the speed of tensor operations with a RandumbTensor vs normal tensors.
    size is .numel() of the rt, so the noise matrix is (size, m) shaped
    `m` being the intermediate dimension of the rt.
    """
    quantiles = [0.5, 0.2, 0.8]
    m = round(size * 0.01)
    x_shape = (int(size//(size**0.5)), int(size**0.5))
    print(f"benching m {m} size {size} shape {x_shape}")
    
    rt_coef = torch.randn(m, requires_grad=True, device=DEVICE)
    rt = CreateRandumbTensor(rt_coef, 1337, x_shape, None, False)
    
    t = torch.randn(x_shape, device=DEVICE)
    mat1 = torch.randn(x_shape, device=DEVICE).T
    
    def rt_impl():
        mat1 @ rt
        return
    
    def rt_fused_impl():
        rt_mm(mat1, rt._coef, rt.size(), rt.stride(), rt.seed, rt.scale_bias, rt.fused_bias)
        return
    
    def torch_impl():
        mat1 @ t
        return
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_impl, quantiles=quantiles)
        
    if provider == 'rt_fused':
        ms, min_ms, max_ms = triton.testing.do_bench(rt_fused_impl, quantiles=quantiles)
        
    if provider == 'rt':
        ms, min_ms, max_ms = triton.testing.do_bench(rt_impl, quantiles=quantiles)
    
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(4, 17, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch_view', 'rt_view'],  # Possible values for `line_arg`.
        line_names=['torch_view', 'rt_view'],  # Label name for the lines.
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('purple', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='torch vs RT viewops speed performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_speed_rt_viewops(size, provider):
    """
    Benchmarks the speed of tensor operations with a RandumbTensor vs normal tensors.
    `m` being the intermediate dimension of the rt.
    """
    quantiles = [0.5, 0.2, 0.8]
    m = 16
    # intentionally make it not **2
    x_shape = (size, 4, 4)
    
    rt_coef = torch.randn(m, requires_grad=True, device=DEVICE)
    rt = CreateRandumbTensor(rt_coef, 1337, x_shape)
    rt_viewed = rt.view(-1, 8, 2)
    
    t = torch.randn(x_shape, device=DEVICE)
    mat1 = torch.randn(size, 4, 4, device=DEVICE)
    mat2 = mat1.view(-1, 2, 8)
    
    def rt_impl():
        rt @ mat1
        return
    
    def torch_impl():
        t @ mat1
        return

    def rt_view_impl():
        rt_viewed @ mat2
        return
    
    def torch_view_impl():
        # for fair comparison, since the RT applies view ops every time it's called, we call view op here instead of caching it
        t.view(-1, 8, 2) @ mat2
        return
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_impl, quantiles=quantiles)
        
    if provider == 'rt':
        ms, min_ms, max_ms = triton.testing.do_bench(rt_impl, quantiles=quantiles)
        
    if provider == 'torch_view':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_view_impl, quantiles=quantiles)
        
    if provider == 'rt_view':
        ms, min_ms, max_ms = triton.testing.do_bench(rt_view_impl, quantiles=quantiles)
        
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(8, 12, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch_rand', 'rt'],  # Possible values for `line_arg`.
        line_names=['torch_rand', 'rt'],  # Label name for the lines.
        styles=[('green', '-'), ('red', '-')],  # Line styles.
        ylabel='MiB',  # Label name for the y-axis.
        plot_name='custom prng peak memory usage',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_mem_bwd(size, provider):
    quantiles = [0.5, 0.2, 0.8]
    inner_dim = 16
    rt_coef = torch.randn(inner_dim, requires_grad=True, device=DEVICE)
    rt = CreateRandumbTensor(rt_coef, 1337, (size,))
    
    noise_mat = torch.randn((size, inner_dim), device=DEVICE)
    coefs = torch.randn(inner_dim, device=DEVICE, dtype=torch.float32, requires_grad=True)
    mat1 = torch.randn((size,), device=DEVICE)

        
    def rt_test():
        y2 = rt * mat1
        y2.backward(torch.randn_like(y2))
        return y2
    
    def torch_rand():
        out = noise_mat @ coefs
        y1 = out * mat1
        y1.backward(torch.randn_like(y1))
        return y1
    
    if provider == 'torch_rand':
        mem_50, mem_20, mem_80 = _test_memory(
            torch_rand,
            quantiles=quantiles,
        )
        
    if provider == "rt":
        mem_50, mem_20, mem_80 = _test_memory(
            rt_test,
            quantiles=quantiles,
        )
        
    return mem_20, mem_50, mem_80

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(2, 20, 1)],  # Different possible values for `size`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'splitn'],  # Possible values for `line_arg`.
        line_names=['torch', 'splitn'],  # Label name for the lines.
        styles=[('green', '-'), ('red', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='speed performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_speed_bwd(size, provider):
    """
    Benchmarks the speed of tensor operations with a RandumbTensor vs normal tensors.
    `m` being the intermediate dimension of the rt.
    """
    quantiles = [0.5, 0.2, 0.8]
    
    dcoefs_size, seed = 256, 1337
    dout = torch.ones((size,), dtype=torch.float32, device="cuda")
    
    noise_mat = squirrel5_generate_2d(size, dcoefs_size, seed)
    
    def torch_impl():
        noise_mat.T @ dout
        return
    
    def matmul_splitn_impl():
        # for fair comparison, since the RT applies view ops every time it's called, we call view op here instead of caching it
        materialize_bwd(dout, dcoefs_size, seed)
        return
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_impl, quantiles=quantiles)

    if provider == 'splitn':
        ms, min_ms, max_ms = triton.testing.do_bench(matmul_splitn_impl, quantiles=quantiles)
        
    return ms, max_ms, min_ms


def profile_mem():
    num_elems = 1024
    inner_dim = 16
    rt_coef = torch.randn(inner_dim, requires_grad=True, device=DEVICE)
    rt = CreateRandumbTensor(rt_coef, 1337, (num_elems,))
    
    noise_mat = torch.randn((num_elems, inner_dim), device=DEVICE)
    coefs = torch.randn(inner_dim, device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    #t = torch.randn(x_shape, device=DEVICE, requires_grad=True)
    mat1 = torch.randn((num_elems,), device=DEVICE)    
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        
        for _ in range(5):
            with record_function("baseline fwd"):
                out = noise_mat @ coefs
                y1 = out * mat1
            with record_function("baseline bwd"):
                y1.backward(torch.randn_like(y1))
            
            with record_function("RT fwd"):
                y2 = rt * mat1
            with record_function("RT bwd"):
                y2.backward(torch.randn_like(y2))
    
    print(prof.key_averages().table())
    prof.export_chrome_trace("trace.json")
    
    
def warmup():
    # if not warmup the memory snapshot will include triton autotune 
    num_elems = 1024
    inner_dim = 16
    rt_coef = torch.randn(inner_dim, requires_grad=True, device=DEVICE)
    rt = CreateRandumbTensor(rt_coef, 1337, (num_elems,))
    
    noise_mat = torch.randn((num_elems, inner_dim), device=DEVICE)
    coefs = torch.randn(inner_dim, device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    #t = torch.randn(x_shape, device=DEVICE, requires_grad=True)
    mat1 = torch.randn((num_elems,), device=DEVICE)


    # warmup first
    for _ in range(5):
        out = noise_mat @ coefs
        y1 = out * mat1
        y1.backward(torch.randn_like(y1))
        
        y2 = rt * mat1
        y2.backward(torch.randn_like(y2))


def check_train_loop():
    
    def print_grad_hook(module, grad_input, grad_output):
        print(f"bwd grad hook {module} {grad_input} {grad_output}")
        
    def print_grad_tensor_hook(t):
        print(f"bwd grad hook on tensor {t.grad}")
    
    class SillyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(1, 5)
            self.rt_linear = SillyLinear(5, 3, 15, 1337, False, device="cuda")
            self.linear1.register_full_backward_hook(print_grad_hook)
            self.rt_linear.register_full_backward_hook(print_grad_hook)
            self.rt_linear.weight._coef.register_post_accumulate_grad_hook(print_grad_tensor_hook)
            
        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.rt_linear(x)
            print(f"fwd out: {x}")
            return x
    
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000).view(2000, 1).to(DEVICE)
    y = y.to(DEVICE)

    # Construct our model by instantiating the class defined above
    model = SillyModel().to(DEVICE)

    print("parameters optimized:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined
    # with torch.nn.Parameter) which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(3):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        # getBack(loss.grad_fn)
        optimizer.zero_grad()
        print(f"loop {t} loss {loss}: out {y_pred} true {y}")
        print("bwd called")
        loss.backward()
        optimizer.step()

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


def plot_dist():
    # plots the distributions of the 3 matrices involved in RandumbTensor, the final output, the noise matrix, and the coef matrix
    import lovely_tensors as lt
    lt.monkey_patch()
    
    """
    out_shape = (128, 9216)
    inner_dim = 64
    torch.manual_seed(20)
    rt_coef = nn.Parameter(torch.randn(inner_dim, requires_grad=True, device=DEVICE), requires_grad=True)
    rt = CreateRandumbTensor(rt_coef, 1337, out_shape)
    """
    
    # rtlinear = SillyLinear(9216, 128, int_dim=64, seed=14, device="cuda")
    rtlinear = SillyLinear(128, 64, int_dim=64, seed=14, device="cuda")
    rt = rtlinear.weight
    rt_coef = rtlinear.weight._coef
    
    y = rt.get_materialized()
    noise = rt.get_noise_matrix()
    print(f"rt materialized: {y.deeper()}")
    print(f"noise: {noise.v}")
    y = y.cpu().detach().numpy().flatten()
    
    noise = noise.cpu().detach().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(noise[:150], cmap='hot', interpolation='nearest')
    fig.savefig('dists/noise_heatmap.png')
    noise = noise.flatten()
    
    coef = rt_coef.cpu().detach().numpy()
    
    print("plotting")
    
    fig, axs = plt.subplots(3)
    fig.suptitle('materialized matrix distribution')
    axs[0].hist(y, bins='auto')
    axs[1].scatter(y, range(y.size))
    axs[2].plot(np.sort(y))
    fig.savefig('dists/materialized_distribution.png')
    
    fig, axs = plt.subplots(3)
    fig.suptitle('noise matrix distribution')
    axs[0].hist(noise, bins=20)
    axs[1].scatter(noise, range(noise.size))
    axs[2].plot(np.sort(noise))
    fig.savefig('dists/noise_distribution.png')
    
    
    fig, axs = plt.subplots(3)
    fig.suptitle('coefs distribution')
    axs[0].hist(coef, bins='auto')
    axs[1].scatter(coef, range(coef.size))
    axs[2].plot(np.sort(coef))
    fig.savefig('dists/coef_distribution.png')



if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Hack for a triton bug - https://github.com/pytorch/pytorch/issues/124565
    torch.empty(1, device='cuda', requires_grad=True).backward()
    
    # warmup()
    torch.cuda.empty_cache()
    # check that the materialization is actually correct
    # check_func()
    # profile_mem()
    # benchmark_mem_bwd.run(print_data=True, show_plots=True, save_path='bench_out')
    
    # benchmarks the speed and memory consumption
    # benchmark_mem_materialize.run(print_data=True, show_plots=True, save_path='bench_out')
    # benchmark_speed_materialize.run(print_data=True, show_plots=True, save_path='bench_out')
    benchmark_speed_rt.run(print_data=True, show_plots=True, save_path='bench_out')
    # benchmark_speed_bwd.run(print_data=True, show_plots=True, save_path='bench_out')

    # check_train_loop()
    # plot_dist()
    