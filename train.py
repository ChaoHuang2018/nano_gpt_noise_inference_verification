"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch_npu 
from torch_npu.contrib import transfer_to_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from rich import print
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 1
eval_only = True # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# Set default dtype globally to float64 (CPU tensors will default to this)
# torch.set_default_dtype(torch.float64)

compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
ctx = torch.cuda.amp.autocast(enabled=False)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split, use_random = True):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    
    if use_random:
        ix = torch.randint(len(data), (batch_size,))
    else:
        # 用固定等间距位置作为 batch（例如开头的几个位置）
        # 保证 ix 长度为 batch_size
        step = max(len(data) // batch_size, 1)
        ix = torch.arange(0, step * batch_size, step)[:batch_size]
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
dtype = 'float32'
scaler = torch.cuda.amp.GradScaler(enabled=False)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

    
model_cpu = GPT(gptconf).to("cpu")

input_error=0 #1e-7
operator_noise=1e-7

noise_info = {
    'input_error': input_error,
    'operator_noise': operator_noise,
    'batch_size': batch_size,
}

#### 统计错误次数
from collections import defaultdict
from datetime import datetime

def analyze_ibp_results_multiple_samples(
    infos_GPU_list,
    infos_CPU_list,
    noise_info=noise_info,
    output_file_prefix="ibp_error_report"
):
    """
    infos_GPU_list: List of infos_GPU, each from a single sample
    infos_CPU_list: List of infos_CPU, each from a single sample
    noise_info: dict with keys 'input_error', 'operator_noise', 'batch_size'
    """

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{output_file_prefix}_{timestamp}.txt"

    layer_stats = defaultdict(lambda: {
        "count": 0,
        "errors": 0,
        "max_outside": 0.0,
        "max_error": 0.0,
        "worst_case_info": None  # (sample_index, layer_index)
    })

    for sample_i, (infos_GPU, infos_CPU) in enumerate(zip(infos_GPU_list, infos_CPU_list)):
        for layer_i, (layer_gpu, layer_cpu) in enumerate(zip(infos_GPU, infos_CPU)):
            layer_type = layer_cpu['layer_type']
            layer_stats[layer_type]["count"] += 1

            y_gpu = layer_gpu['y'].cpu()
            y_cpu = layer_cpu['y'].cpu()
            y_lower = layer_cpu['y_lower'].cpu()
            y_upper = layer_cpu['y_upper'].cpu()

            # 检查是否越界
            is_bounded = (y_gpu >= y_lower).all() and (y_gpu <= y_upper).all()
            if not is_bounded:
                layer_stats[layer_type]["errors"] += 1

                real_minus_lower = y_lower - y_gpu
                real_minus_upper = y_gpu - y_upper
                real_outside = torch.clamp(torch.maximum(real_minus_lower, real_minus_upper), min=0.0)
                outside_max = real_outside.max().item()
                true_error = (y_cpu - y_gpu).abs().max().item()

                if outside_max > layer_stats[layer_type]["max_outside"]:
                    layer_stats[layer_type]["max_outside"] = outside_max
                    layer_stats[layer_type]["max_error"] = true_error
                    layer_stats[layer_type]["worst_case_info"] = (sample_i, layer_i)

    # 写入文本
    lines = []
    lines.append(f"IBP Error Report (Multiple Samples)\n")
    lines.append(f"Generated at: {timestamp}\n")
    lines.append("=" * 60 + "\n")

    # 打印 noise 信息
    if noise_info is not None:
        lines.append("Noise Info:\n")
        for k, v in noise_info.items():
            lines.append(f"  {k}: {v}\n")
        lines.append("-" * 60 + "\n")

    # 分层统计信息
    for layer_type, stats in layer_stats.items():
        count = stats["count"]
        errors = stats["errors"]
        freq = errors / count if count > 0 else 0

        lines.append(f"Layer Type     : {layer_type}\n")
        lines.append(f"  Total Count  : {count}\n")
        lines.append(f"  Error Count  : {errors}\n")
        lines.append(f"  Error Rate   : {freq:.4f}\n")
        if errors > 0:
            lines.append(f"  Max Outside Error  : {stats['max_outside']:.6e}\n")
            lines.append(f"  True CPU/GPU Error : {stats['max_error']:.6e}\n")
            sample_i, layer_i = stats["worst_case_info"]
            lines.append(f"  Location (Sample #{sample_i}, Layer #{layer_i})\n")
        lines.append("-" * 60 + "\n")

    # 写入到父路径
    parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_path = os.path.join(parent_path, output_file_name)

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"[✔] IBP多样本错误统计已写入: {output_path}")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    model_cpu.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(1):
            
            # npu output
            X, Y = get_batch(split, use_random=False)
            
            infos_GPU_list = []
            infos_CPU_list = []
            
            for sample_idx in range(batch_size):
                
                print("-------------", str(sample_idx), " starts. ----------------")
                print("------------------------------------------------------")
                
                X_idx = X[sample_idx:sample_idx+1]  # 测试Batch第sample_idx个样本
                Y_idx = Y[sample_idx:sample_idx+1]
            
                # robust_veri can be Off, CPU, GPU
                infos_GPU = []
                infos_cpu = []
                with ctx:
                    print("GPU is now producing layer-wise outputs!")
                    logits, loss, infos = model(X_idx, Y_idx, robust_veri='GPU', input_error=input_error, infos_GPU=infos_GPU, noising_input=False, operator_noise=operator_noise)
                losses[k] = loss.item() 
                print("GPU layer-wise outputs are all generated!")
                # print(len(infos_GPU))

                # cpu lower upper bound
                model_cpu.load_state_dict(model.state_dict())
                # model_cpu.to("cpu")
                # print(model.training)      # True 或 False
                # print(model_cpu.training)  # True 或 False
                # exit()
                print("CPU is now producing layer-wise output ranges!")
                logits_cpu, loss_cpu, infos_cpu = model_cpu(X_idx.to("cpu"), Y_idx.to("cpu"), robust_veri='CPU', input_error=input_error, infos_GPU=infos_GPU, noising_input=False, operator_noise=operator_noise)
                for layer_i in range(len(infos_GPU)):
                    # print(infos["x"][layer_i].to("cpu") >= infos_cpu["x_lower"][layer_i])
                    # print(infos["x"][layer_i].to("cpu") <= infos_cpu["x_upper"][layer_i])
                    y_gpu = infos_GPU[layer_i]['y'].cpu()
                    y_cpu = infos_cpu[layer_i]['y'].cpu()
                    y_lower = infos_cpu[layer_i]['y_lower'].cpu()
                    y_upper = infos_cpu[layer_i]['y_upper'].cpu()
                    is_bounded = (y_gpu >= y_lower).all() and (y_gpu <= y_upper).all()
                    # print(str(k) + "-th input, " + str(noise_number) + "-th random noise, " + str(layer_i) + "-th layer, " + str(is_bounded))
                    print(str(k) + "-th input, " + str(layer_i) + "-th layer, " + infos_cpu[layer_i]['layer_type'] + ": "+ str(is_bounded))
                # is_bounded = is_bounded.any()
                    if is_bounded == False:
                        real_minus_lower = y_lower - y_gpu
                        real_minus_upper = y_gpu - y_upper
                        real_outside = torch.maximum(real_minus_lower, real_minus_upper)
                        real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
                        print(infos_cpu[layer_i]['layer_type'], ". Output range width: ", (y_upper - y_lower).max().item())
                        print(infos_cpu[layer_i]['layer_type'], ". GPU output range error: ", real_outside.max().item())
                        print(infos_cpu[layer_i]['layer_type'], ". Real GPU/CPU output error: ", (y_cpu - y_gpu).max().item())
                infos_GPU_list.append(infos_GPU)
                infos_CPU_list.append(infos_cpu)
            analyze_ibp_results_multiple_samples(infos_GPU_list, infos_CPU_list)
            exit()  
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
#     for micro_step in range(gradient_accumulation_steps):
#         if ddp:
#             # in DDP training we only need to sync gradients at the last micro step.
#             # the official way to do this is with model.no_sync() context manager, but
#             # I really dislike that this bloats the code and forces us to repeat code
#             # looking at the source of that context manager, it just toggles this variable
#             model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
#         X, Y = get_batch('train')
#         with ctx:
#             logits, loss, infos = model(X, Y)
#             loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
#         # immediately async prefetch next batch while model is doing the forward pass on the GPU
        
#         # backward pass, with gradient scaling if training in fp16
#         # scaler.scale(loss).backward()
#         loss.backward()
#         print(f"[Train step loss] {loss.item()}")
#         # for name, param in model.named_parameters():
#         #     if param.grad is not None:
#         #         print(name, param.grad.abs().mean())
#     # clip the gradient
#     if grad_clip != 0.0:
#         # scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#     # step the optimizer and scaler if training in fp16
#     # scaler.step(optimizer)
#     # scaler.update()
#     optimizer.step()
#     # flush the gradients as soon as we can, no need for this memory anymore
#     optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    
    # print("iter_num: " + str(iter_num))
    # print("max_iters: " + str(max_iters))
    

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
