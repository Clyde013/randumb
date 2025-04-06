"""
This file was adapted from the modded-nanogpt speedrunning repo record #14 (https://github.com/KellerJordan/modded-nanogpt/blob/master/records/120424_ValueEmbed/train_gpt2.py), 
which introduced value embeddings but predates many of YouJiaCheng's insane cooking, which I feel is a good tradeoff point between the speedrunning hyperoptimizations while
still remaining a reasonably representative benchmark as far as nanoGPT training goes. Also turns off coordinate_descent_tuning.

Includes WandB integration which original script does not use. 
"""


import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import contextlib
from dataclasses import dataclass
from typing import Optional, Dict

import wandb
import numpy as np
import torch

import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("..")
from src.SillyLayers import stepgen

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8 # batch size, in sequences, across all devices
    sequence_length : int = 64*1024 # sequence length, in tokens
    num_iterations : int = 1530 # number of iterations to run
    warmup_iters : int = 0
    cooldown_iters : int = 600 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    wandb_integration : bool = True # enable logging to wandb
    single_gpu_training: bool = True # modifies batch_size and seq_len to fit on single 4090
    normal_model: bool = False # whether its a normal or random model
    # seed generator params
    offset: int = 13
    stepsize: int = 7
    int_dim: int = 256
args = Hyperparameters()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model
if args.normal_model:
    from gpt_models import GPT, GPTConfig, CastedLinear
else:
    from gpt_models import RT_GPT as GPT, RT_GPTConfig as GPTConfig, RT_CastedLinear as CastedLinear

# NOTE: the expected starting loss should be ln(num_vocab) = 10.8258

if args.wandb_integration:
    run = wandb.init(
        project="nanogpt",  # Specify your project
        config=vars(args),  # convert args dataclass into dict and pass in all the configs
    )

if args.single_gpu_training:
    # ref: https://github.com/KellerJordan/modded-nanogpt/pull/38 to support 4090 training
    # per https://github.com/KellerJordan/modded-nanogpt/pull/38#issuecomment-2571392240
    # we halve the sequence length and double the batch size
    args.batch_size = args.batch_size * 8
    args.sequence_length = args.sequence_length // 8
    print(f"single gpu training active, adjusting batch size to {args.batch_size} and seq len to {args.sequence_length}")
    
    
# if run with CUDA_VISIBLE_DEVICES=0 should work for single GPU
# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"ddp_rank {ddp_rank} ddp_local_rank {ddp_local_rank} ddp_world_size {ddp_world_size} device {device}")
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
    
device_type = 'cuda' if 'cuda' in device else 'cpu'

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write(code)
        f.write('='*100 + '\n')
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# convenience variables
T = args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (T * ddp_world_size) == 0
val_steps = args.val_tokens // (T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size

# load tokens
train_loader = DistributedDataLoader(args.input_bin, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
if args.normal_model:
    conf = GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)
else:
    # sort of hardcoded the int_dim, constant 128 across all layers
    int_dim_gen = stepgen(args.int_dim, 0)
    seed_gen = stepgen(args.offset, args.stepsize)
    conf = GPTConfig(int_dim_gen=int_dim_gen, seed_gen=seed_gen, vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)
    
model = GPT(conf)
# model = model.to(device).to(torch.bfloat16)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        # m.to(torch.float32)
        m.float()

# print("MODEL:")
# for name, m in model.named_modules():
#     print(name, m)

# THIS IS TURNED OFF WHY DOES IT INCREASE COMPILE TIMES BY 25+ MINS FOR A FEW SECS SPEEDUP
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = False # suggested by @Chillee
    
# torch compile bugged asf, will compile a model that memleaks
if args.normal_model: model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# init the optimizer(s)
optimizer1 = torch.optim.Adam([*raw_model.transformer.wte.parameters(), *raw_model.transformer.vte.parameters()], lr=0.6, betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([*raw_model.lm_head.parameters()], lr=0.008, betas=(0.8, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
if len(matrix_params) == 0:
    # this happens if random model, since our coef matrices are all 1D
    optimizer3 = None
    optimizers = [optimizer1, optimizer2, optimizer4]
else:
    optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
    attn_blocksize = torch.tensor(64*((step/args.num_iterations * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
            
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        if args.wandb_integration: wandb.log({"val_step": step, "val_loss": val_loss.item()})
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        ctx = model.no_sync() if i < train_accumulation_steps else contextlib.nullcontext()
        with ctx: # there's no need to sync gradients every accumulation step
            # forward pass
            loss = model(x, y, attn_blocksize=attn_blocksize)
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()
        train_loss = loss.detach()
        
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    if optimizer3 is not None:
        frac = min(step/300, 1)
        optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
    if args.wandb_integration: wandb.log({"step": step+1, "train_loss": train_loss.item(), "train_time":approx_time, "step_avg":approx_time/timed_steps})

if master_process:
    
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()