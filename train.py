"""GPT-2 on FineWeb-edu with DDP or FSDP2, for the fault-tolerance measurements.

Run it with torchrun. The settings are environment variables:

    MODE      ddp or fsdp
    MODEL     124M or 1.5B
    STEPS     optimizer steps
    BATCH     tokens per step, across all GPUs
    MICRO     sequences per micro-batch per GPU
    CKPT_ACT  1 to recompute the activations of each block in the backward pass
    COMPILE   0 to run without torch.compile
    WARMUP    steps of linear learning-rate warmup
    PROFILE   steps to profile at the end, for the communication time
    DATA      the folder with the llm.c token shards from data.py
    CKPT      the folder for checkpoints; a run resumes from the latest one in it
    CKPT_EVERY  steps between checkpoints, 0 for none
    ASYNC     1 to save with dcp.async_save

Every world size reads the same tokens at each step, in the same micro-batches,
so the loss of a run on 4 GPUs can be compared with the loss on 1 GPU.
The last line is `result {...}` with the medians of the timed steps.
"""

import glob
import json
import math
import os
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

SIZES = {"124M": dict(layers=12, heads=12, dim=768), "1.5B": dict(layers=48, heads=25, dim=1600)}
VOCAB = 50304  # 50257 padded to a multiple of 128
SEQ = 1024
# Peak dense bf16 FLOP/s per GPU, from the NVIDIA data sheets. Modal's H100 is the SXM part.
# The first name that is in the device name wins, so L40S comes before L4.
PEAK = {"L40S": 362e12, "L4": 121e12, "H100": 989e12}
WARM = 5  # steps left out of the medians, for the compile


def env(name: str, default):
    return type(default)(os.environ.get(name, default))


class Block(nn.Module):
    def __init__(self, heads: int, dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc, self.out = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).transpose(1, 3).unbind(2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, layers: int, heads: int, dim: int, ckpt_act: bool) -> None:
        super().__init__()
        self.ckpt_act = ckpt_act
        self.wte, self.wpe = nn.Embedding(VOCAB, dim), nn.Embedding(SEQ, dim)
        self.blocks = nn.ModuleList(Block(heads, dim) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB, bias=False)
        self.head.weight = self.wte.weight
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif p.dim() > 1:
                # GPT-2 scales the residual projections by the depth.
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.ckpt_act else block(x)
        logits = self.head(self.ln(x))
        return F.cross_entropy(logits.float().view(-1, VOCAB), targets.view(-1))


def shard(path: str) -> np.ndarray:
    """The tokens of an llm.c shard: a 256 x int32 header, then uint16 tokens."""
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4)


def micro_batches(tokens: np.ndarray, start: int, count: int, micro: int, rank: int, world: int):
    """Micro-batches rank, rank + world, ... of the `count` that start at token `start`."""
    n = micro * SEQ
    for m in range(rank, count, world):
        buf = torch.from_numpy(tokens[start + m * n:start + (m + 1) * n + 1].astype(np.int64))
        yield buf[:-1].view(micro, SEQ).cuda(non_blocking=True), buf[1:].view(micro, SEQ).cuda(non_blocking=True)


def union(spans: list[tuple[float, float]]) -> list[list[float]]:
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def comm_times(trace: str) -> tuple[float, float]:
    """The NCCL kernel time in a Chrome trace, and the part of it that no other kernel overlaps."""
    kernels = [e for e in json.load(open(trace))["traceEvents"] if e.get("cat") == "kernel"]
    comm = union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" in e["name"].lower()])
    comp = union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()])
    total = sum(e - s for s, e in comm)
    overlap, j = 0.0, 0
    for s, e in comm:
        while j < len(comp) and comp[j][1] <= s:
            j += 1
        k = j
        while k < len(comp) and comp[k][0] < e:
            overlap += min(e, comp[k][1]) - max(s, comp[k][0])
            k += 1
    return total / 1e3, (total - overlap) / 1e3  # microseconds to milliseconds


def latest(folder: str) -> int:
    """The step of the newest complete checkpoint in `folder`, or 0.

    DCP writes .metadata last, after every rank has written its part.
    """
    done = glob.glob(os.path.join(folder, "step_*", ".metadata"))
    return max((int(p.split(os.sep)[-2][5:]) for p in done), default=0)


def main() -> None:
    mode, size = env("MODE", "ddp"), env("MODEL", "124M")
    steps, batch, micro = env("STEPS", 50), env("BATCH", 262144), env("MICRO", 8)
    profile = env("PROFILE", 0)
    ckpt, every = env("CKPT", ""), env("CKPT_EVERY", 0)
    # async_save stages the state on the CPU and needs a CPU process group.
    dist.init_process_group("cuda:nccl,cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    count = batch // (micro * SEQ)
    assert count * micro * SEQ == batch and count % world == 0, "BATCH must split into MICRO x SEQ x world"
    log = print if rank == 0 else (lambda *a, **k: None)

    torch.manual_seed(0)
    cfg = SIZES[size]
    model = GPT(**cfg, ckpt_act=env("CKPT_ACT", 0) == 1).cuda()
    n_params = sum(p.numel() for p in model.parameters()) - model.wpe.weight.numel()
    flops_per_token = 6 * n_params + 12 * cfg["layers"] * cfg["dim"] * SEQ
    if env("COMPILE", 1):
        for block in model.blocks:
            block.compile()
    if mode == "fsdp":
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        train = model
    else:
        train = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    warmup = min(env("WARMUP", 20), steps)

    def lr(s: int) -> float:
        cos = 0.5 * (1 + math.cos(math.pi * min(1, max(0, s - warmup) / max(1, steps - warmup))))
        return 6e-4 * min((s + 1) / warmup, 0.1 + 0.9 * cos)

    start, saving = latest(ckpt) if ckpt else 0, None
    if start:
        t0 = time.perf_counter()
        model_sd, opt_sd = get_state_dict(train, opt)
        state = dict(model=model_sd, optim=opt_sd)
        dcp.load(state, checkpoint_id=os.path.join(ckpt, f"step_{start:06d}"))
        set_state_dict(train, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        log(f"ckpt load step {start} {(time.perf_counter() - t0) * 1e3:.0f} ms")

    shards = [shard(p) for p in sorted(glob.glob(os.path.join(env("DATA", "data"), "edu_fineweb_train_*.bin")))]
    per_shard = (len(shards[0]) - 1) // batch
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK.items() if k in gpu), None)
    log(f"{mode} {size} on {world} x {gpu}: {n_params / 1e6:.0f}M parameters, "
        f"{count // world} micro-batches of {micro} per GPU per step")

    times, losses = [], []
    prof = None
    for step in range(start, steps):
        if profile and step == steps - profile:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            prof.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = shards[(step // per_shard) % len(shards)]
        total = torch.zeros((), device="cuda")
        batches = list(micro_batches(tokens, (step % per_shard) * batch, count, micro, rank, world))
        for i, (x, y) in enumerate(batches):
            last = i == len(batches) - 1
            if mode == "fsdp":
                train.set_requires_gradient_sync(last)
            with (train.no_sync() if mode == "ddp" and not last else torch.enable_grad()):
                with torch.autocast("cuda", torch.bfloat16):
                    loss = train(x, y) / count
                loss.backward()
            total += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(total)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if prof is None:
            times.append(dt)
        losses.append(total.item())
        mfu = flops_per_token * batch / dt / (peak * world) if peak else float("nan")
        norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
        if saving and saving[1].done():
            log(f"ckpt saved step {saving[0]} {(time.perf_counter() - saving[2]) * 1e3:.0f} ms")
            saving = None
        if every and (step + 1) % every == 0:
            if saving:
                saving[1].result()
                log(f"ckpt saved step {saving[0]} {(time.perf_counter() - saving[2]) * 1e3:.0f} ms")
            t1 = time.perf_counter()
            model_sd, opt_sd = get_state_dict(train, opt)
            path = os.path.join(ckpt, f"step_{step + 1:06d}")
            if env("ASYNC", 0):
                saving = (step + 1, dcp.async_save(dict(model=model_sd, optim=opt_sd), checkpoint_id=path), t1)
            else:
                dcp.save(dict(model=model_sd, optim=opt_sd), checkpoint_id=path)
            log(f"ckpt save step {step + 1} blocked {(time.perf_counter() - t1) * 1e3:.0f} ms")
        log(f"step {step + 1}/{steps} loss {losses[-1]:.4f} norm {norm.item():.3f} {dt * 1e3:.0f} ms "
            f"{batch / dt:.0f} tok/s mfu {mfu:.1%} mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    if saving:
        saving[1].result()
        log(f"ckpt saved step {saving[0]} {(time.perf_counter() - saving[2]) * 1e3:.0f} ms")
    result = dict(start=start, mode=mode, model=size, world=world, gpu=gpu, steps=steps, batch=batch, micro=micro,
                  params=n_params, loss=losses[-1], losses=losses)
    timed = times[WARM:]
    if timed:
        dt = statistics.median(timed)
        result.update(step_ms=dt * 1e3, tok_s=batch / dt,
                      mfu=flops_per_token * batch / dt / (peak * world) if peak else None)
    mem = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device="cuda")
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    result["peak_mem_gb"] = mem.item()
    if prof is not None:
        prof.stop()
        trace = f"/tmp/trace_{rank}.json"
        prof.export_chrome_trace(trace)
        comm, exposed = (torch.tensor(v / profile, device="cuda") for v in comm_times(trace))
        for t in (comm, exposed):
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
        result.update(comm_ms=comm.item(), exposed_comm_ms=exposed.item())
    log("result " + json.dumps(result))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
