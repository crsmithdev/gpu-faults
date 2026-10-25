# To do

Things to build or look at in gpu-faults. `aleph todo` owns this file.
Item numbers never change: commits, the spec and the vault quote them.

## 1. DDP vs FSDP2 on 4 GPUs, checkpoint cost, and recovery from a crashed, hung or lost worker
---
id: 1
status: done
created: 2026-10-22
updated: 2026-10-25
priority: high
labels: [project, modal]
---

Measure MFU, peak memory per GPU, exposed communication time, and loss parity against 1 GPU.
Then kill one rank. Measure recovery time, and lost work against checkpoint overhead as the checkpoint interval changes.

### Notes

- 2026-10-22 10:13: 2026-10-22: Chris picked this as the next project and post, over the GPT-2 speedrun (#1).
- 2026-10-22 15:30: Scope decided 2026-10-22: 4×L4 in one container (PCIe, no NVLink) for all runs, plus one 4×H100 confirmation. Faults: crash (SIGKILL one rank), hang (SIGSTOP one rank, NCCL watchdog), lost container (needs modal-lib retries in the recipe and a run record that survives a second container). Models: GPT-2 124M for scaling and fault runs; 1.5B to show DDP OOM on 24 GB L4 against FSDP2 fitting across 4. Sections: DDP vs FSDP2; DCP checkpoint cost (sync vs async_save); fault timelines; checkpoint interval against Young/Daly; limits. Budget ~$20–35 on L4 plus ~$5 on H100. Launcher torchrun --max-restarts as the recipe cmd; DCP checkpoints on the /runs Volume.
- 2026-10-22 19:17: 2026-10-22: DDP/FSDP2 comparison, crash, hang and lost-container recovery measured; see docs/findings.md. Left: checkpoint cost by size and sync vs async, checkpoint interval against Young/Daly, 1.5B DDP OOM (L4 call pending), H100 check, README, post.
- 2026-10-22 20:35: 2026-10-23 UTC: checkpoint cost measured; 1.5B DDP OOM on 4×L4 confirmed; found two kinds of 4×L40S host (FSDP2 124M 660 vs 1030 ms per step, cause unconfirmed); see docs/findings.md (60ebe7b). Post draft on crsmithdev.com branch post/gpu-faults (ddb34c7). Left: checkpoint interval vs Young/Daly, H100 check, README, finish post.
- 2026-10-22 21:09: 2026-10-23 04:08 UTC: checkpoint interval (best 10 steps, Young 8.3 s) and H100 runs done; README written; findings 27238c0. Post draft 1f77949 on branch post/gpu-faults is complete apart from Chris's review. Spend $20.73.
- 2026-10-25 08:33: 2026-10-25: post published at https://crsmith.dev/blog/gpu-faults/ (crsmithdev.com b86e4bf); repo made public.
