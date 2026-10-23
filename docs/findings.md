# Findings

Measurements from the runs on Modal, 2026-10-22. GPT-2 124M on FineWeb-edu,
262,144 tokens per step, unless a section says otherwise. The run IDs are
on the Volume `faults-jobs-runs`; `uv run mlib logs <id>` shows a log.

## Queue time

| GPU | Call to container start |
| - | - |
| 4×L4 | 1752, 2662, 4065 and 5850 s |
| 4×L40S | 6 to 716 s |

4×L4 queues for 30 to 100 minutes, so the runs use 4×L40S.

## DDP and FSDP2, 200 steps

| Run | Step | MFU | Peak memory per GPU | NCCL time (total / not hidden) | Loss gap to 1 GPU (mean / max) |
| - | - | - | - | - | - |
| DDP, 4×L40S | 738 ms | 21.0% | 9.77 GB | 159 / 118 ms | 0.010 / 0.030 |
| FSDP2, 4×L40S | 1028 ms | 15.1% | 7.83 GB | 696 / 257 ms | 0.019 / 0.064 |
| DDP, 1×L40S | 2518 ms | 24.6% | 9.77 GB | none | baseline |
| FSDP2 1.5B, 4×L4, 65,536 tokens per step | 18,976 ms | 7.4% | 13.5 GB | 17,941 / 11,744 ms | none |

Runs: fc-01M5K67PQ7P9MRP8DZS7W2M04A (DDP), fc-01M5K67PHTEFSP65B8YPN2QR8M
(FSDP2), fc-01M5K48MFK4PVYM7C4KKPCZ251 (1 GPU), fc-01M5K48K5MMPW5D3X7Z4JWNQ63
(1.5B). DDP with 1.5B on 4×L4 ran out of memory before step 1: 21.44 GiB
allocated of 22.06 GiB (fc-01M5K48K295SYDT97YPHZJVSJV). MFU is against the dense bf16 peak: 362 TFLOP/s for the L40S and
121 for the L4.

## Faults

FSDP2 on 4×L40S, a checkpoint every 20 steps, the fault at step 50 or 70.

| Fault | Detection | Workers start | Checkpoint loaded | Back at the fault step |
| - | - | - | - | - |
| Crash (SIGKILL rank 3) | 0.9 s | 4.1 s | 9.1 s | 21.7 s |
| Hang (SIGSTOP rank 3), NCCL watchdog only | about 1400 s | none: the job failed | none | none |
| Hang, step watchdog of 30 s | 30.3 s | 34.4 s | 39.5 s | 52.1 s |
| Lost container (SIGKILL PID 1), one Modal retry | none | 15.5 s | 21.1 s | 33.8 s |

Runs: fc-01M5K667BZ951Y07EV5TC477WS (crash), fc-01M5K4BK5KE6S4G22G1P252J7Y
(hang, NCCL_TIMEOUT=60), fc-01M5K7QMWG98Y1D8H7TMBQ31P1 (hang, step
watchdog), fc-01M5K7YT7CZMA7F6PX6YGTCG1S (lost container).

- A hung rank keeps the NCCL watchdog from ending the job. The watchdog
  finds the stuck collective at NCCL_TIMEOUT, but the abort blocks on the
  stopped peer. The heartbeat monitor ends the processes about 23 minutes
  after the fault, with the default timeout or with 60 s. That costs about
  $3 of 4×L40S time per hang.
- The step watchdog must be a separate process. The training loop waits in
  `.item()`, which holds the GIL, so a watchdog thread never runs.
- After a torchrun restart in torch 2.11, a gloo process group fails with
  "Connection refused". NCCL alone comes back. `async_save` needs a gloo
  group, so the fault runs save synchronously.
- A checkpoint on the runs Volume survives a lost container only after the
  next modal-lib heartbeat commits the Volume, up to 30 s after the save.

## Two kinds of L40S host

The same FSDP2 124M run takes 1011 to 1060 ms per step on most 4×L40S
containers, and 657 to 671 ms on some. 3 of about 30 FSDP2 runs got the fast
kind, all in one batch launched at 02:55 UTC (fc-01M5KAQ6V431NMJT6G8RMVQFVS,
fc-01M5KAQ5H4ES4PKJW9W4JAPN6M, fc-01M5KAQ82KHA6QAJERT26M2D4M). The
difference is in the communication: memory and loss are the same.

On 12 slow containers, no GPU pair has peer access
(`torch.cuda.can_device_access_peer`), and NCCL uses `SHM/direct` for every
channel. `nvidia-smi topo -m` fails in the sandbox, and the CPU model is
hidden. The fast containers ran before the diagnostic existed, so the cause
is not confirmed. The guess is P2P over a PCIe switch on the fast hosts.

All the DDP and FSDP2 numbers above are from slow hosts.

## Checkpoint save, FSDP2 on 4×L40S

| Model | Size | Save | Blocks training | Save done |
| - | - | - | - | - |
| 124M | about 1.5 GB | sync, local disk | 0.50 to 0.80 s | same |
| 124M | | sync, runs Volume | 1.00 to 1.42 s | same |
| 124M | | async, local disk | 0.11 to 0.31 s | 6.4 to 7.5 s later |
| 124M | | async, runs Volume | 0.14 to 0.34 s | 7.4 to 7.9 s later |
| 1.5B | about 18.7 GB | sync, local disk | 5.2 to 6.2 s | same |
| 1.5B | | async, local disk | 1.1 to 2.0 s | 5.7 s later; the first save 185 s |

The size is 12 bytes per parameter: fp32 weights and the two AdamW moments.
The first save of a run takes longer. The 1.5B sync run was most likely on a
fast host (5.4 s per step) and the async run on a slow one (18.2 s per step),
so the pair does not show what async saving costs in step time.

Runs: fc-01M5KAQAT59CNA6Y9YBNQRK0KM, fc-01M5KAQ6V431NMJT6G8RMVQFVS,
fc-01M5KAQ5H4ES4PKJW9W4JAPN6M, fc-01M5KAQ82KHA6QAJERT26M2D4M (124M),
fc-01M5KAQ7BAJYPW6JD704DQWPXS, fc-01M5KAQ3WKDQZM56DQTXQEXZ71 (1.5B).

## Spend

$16.57 over 42 runs on `faults-jobs`, to 2026-10-22 03:40 UTC, plus under
$0.10 of CPU probes.
