# Findings

Measurements from the runs on Modal, 2026-10-22. GPT-2 124M on FineWeb-edu,
262,144 tokens per step, unless a section says otherwise. The run IDs are
on the Volume `faults-jobs-runs`; `uv run mlib logs <id>` shows a log.

## Queue time

| GPU | Call to container start |
| - | - |
| 4×L4 | 1752, 2662 and 4065 s; one call still pending after 70 min |
| 4×L40S | 6 to 716 s |

4×L4 queues for 30 to 70 minutes, so the runs use 4×L40S.

## DDP and FSDP2, 200 steps

| Run | Step | MFU | Peak memory per GPU | NCCL time (total / not hidden) | Loss gap to 1 GPU (mean / max) |
| - | - | - | - | - | - |
| DDP, 4×L40S | 738 ms | 21.0% | 9.77 GB | 159 / 118 ms | 0.010 / 0.030 |
| FSDP2, 4×L40S | 1028 ms | 15.1% | 7.83 GB | 696 / 257 ms | 0.019 / 0.064 |
| DDP, 1×L40S | 2518 ms | 24.6% | 9.77 GB | none | baseline |
| FSDP2 1.5B, 4×L4, 65,536 tokens per step | 18,976 ms | 7.4% | 13.5 GB | 17,941 / 11,744 ms | none |

Runs: fc-01M5K67PQ7P9MRP8DZS7W2M04A (DDP), fc-01M5K67PHTEFSP65B8YPN2QR8M
(FSDP2), fc-01M5K48MFK4PVYM7C4KKPCZ251 (1 GPU), fc-01M5K48K5MMPW5D3X7Z4JWNQ63
(1.5B). MFU is against the dense bf16 peak: 362 TFLOP/s for the L40S and
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

## Checkpoint save, 124M on 4×L40S, FSDP2

A synchronous save blocks for 0.5 to 0.8 s to local disk (`/tmp`) and 1.0
to 1.4 s to the runs Volume. The first save of a run takes longer.
