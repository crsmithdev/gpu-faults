# gpu-faults

A GPT-2 training job on 4 GPUs loses a worker. How long until it trains
again, how much work does it lose, and what does the protection cost?
This repo measures it on [Modal](https://modal.com) with DDP and FSDP2,
DCP checkpoints, and injected faults: a crashed rank, a hung rank, and a
lost container.

The results are in [docs/findings.md](docs/findings.md).

## Setup

```sh
uv sync
uv run modal setup   # once
uv run mlib run data # the FineWeb-edu token shards, to the Volume faults-data
```

`modal.json` has the recipes. [modal-lib](https://github.com/crsmithdev/modal-lib)
runs them and records the time and cost of each run. `train.py` documents
its settings, which are environment variables. `--set env.NAME=value`
changes one setting for one run.

The `train` recipe runs `torchrun` on 4×L40S in one container and restarts
the workers up to 3 times. Each run prints the peer access between its GPUs
(`p2p`), because the step time of FSDP2 depends on it.

## Runs

DDP and FSDP2, with the NCCL time of the last 5 steps:

```sh
uv run mlib run train --set env.MODE=ddp --set env.STEPS=200 --set env.PROFILE=5
uv run mlib run train --set env.MODE=fsdp --set env.STEPS=200 --set env.PROFILE=5
uv run mlib run train --set gpu=L40S --set env.STEPS=200   # 1 GPU, for the loss
```

A crash, a hang and a lost container, with a checkpoint every 20 steps:

```sh
uv run mlib run train --set env.MODE=fsdp --set env.STEPS=80 --set env.CKPT=/tmp/ck \
  --set env.CKPT_EVERY=20 --set env.FAULT=crash --set env.FAULT_STEP=50
uv run mlib run train --set env.MODE=fsdp --set env.STEPS=80 --set env.CKPT=/tmp/ck \
  --set env.CKPT_EVERY=20 --set env.FAULT=hang --set env.FAULT_STEP=50 --set env.STEP_TIMEOUT=30
uv run mlib run train --set env.MODE=fsdp --set env.STEPS=100 --set "env.CKPT={run}/ckpt" \
  --set env.CKPT_EVERY=20 --set env.FAULT=container --set env.FAULT_STEP=70 --set retries=1
```

A lost container needs the checkpoint on a Volume. `{run}` is the run
folder on the Volume `faults-jobs-runs`, which modal-lib commits every 30 s.

`uv run mlib ps` lists the runs, and `uv run mlib logs <id>` shows a log.
Each log line starts with its time, so the time of each recovery phase
comes from the lines `fault`, `proc start`, `ckpt load` and `step`.

## Limits

- One machine. Modal's multi-node support is in beta.
- After a torchrun restart, a gloo process group fails in torch 2.11, so
  `ASYNC=1` does not work with a restart.
- A checkpoint on the Volume survives a lost container only after the next
  commit.
