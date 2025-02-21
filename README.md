# RANDUMB: Mostly Random Neural Networks

<b>How random can a model be?</b>

This repo is home to an extremely dumb attempt at incorporating random projections into neural network training to create the silliest architecture possible - a *mostly* random one.

While traditional neural network pretraining techniques may optimize every parameter available in the model, what if we kept a subset of those parameters *completely random*? 
The next logical conclusion of course is that if a large subset of the model were to be completely random, we could simply compress that into a seed and generate the values on the fly (beacuse why not?).
The obvious benefits of this approach would be runtime memory efficiency, as the random tensors would only ever require materialization as necessary - you could in theory utilize a rather *ludicrously* large random matrix without ever needing to store any of those numbers in GPU VRAM, especially if fused with further matrix operations.
Not to mention the final model could be represented with a single seed and a much smaller number of the trainable parameters.

While there has been similar work in the past combining random projections and neural networks, most of the implementations utilize custom hardware (i.e. FPGA, Graphcore IPUs), which obviously aren't widely available. 
This repo implements a custom triton Pseudorandom Matrix-Vector Product kernel utilizing the [Squirrel5 PRNG Noise Function](https://twitter.com/SquirrelTweets/status/1421251894274625536) in order to efficiently represent and materialize custom tensors:

![](readme_src/randumbTensor.png)

It is easy to see how the utilization of a PRNG noise function instead of more traditional sequential PRNGs offers easier parallelization and random access of the noise matrix. 
Most naive SGEMV operations are memory-bound, so one can additionally balance the memory-compute bandwidth requirements of the PMV by adjusting the size of the coefficient dimension.
For further details on the custom kernel implementation, check out the code directly, as well as an attached diagram under `readme_src/materialization_algo.png` to illustrate the algorithm.

We also throw in a fused elementwise normalization factor by the std. to the final materialized tensor as the dot product between two independent uniform (a row of noise: x) and normal (coef: y) distributions will have variance of:
```math
\text{var}(x) = \frac{(b−a)^2}{12} = 1/3 \newline
\text{var}(y) = 1 \newline
\text{Since both X and Y have 0 mean:} \
\text{var}(x \cdot y) = \sum^{d_y}_{i=1}{\text{var}(xy)} = \sum^{d_y}_{i=1}{\text{var}(x)\text{var}(y)} = \sum^{d_y}_{i=1}{\frac{1}{3}} = \frac{d_y}{3}
```

## Requirements
This repo uses triton==3.2.0 and pytorch>=2.6.0 for `torch.library.triton_op` support. Install with:

`pip3 install --pre torch==2.6.0.dev20250104+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124`

Anything else install as you find import errors pop up :)


## Relevant papers (non-exhaustive)

Shafipour, R. et al. (2024) ‘SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators’. Available at: https://doi.org/10.48550/arXiv.2410.10714

Aghajanyan, A., Zettlemoyer, L. and Gupta, S. (2020) ‘Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning’. arXiv. Available at: http://arxiv.org/abs/2012.13255
