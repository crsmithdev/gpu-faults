# gpu-faults

A GPT-2 training job on 4 GPUs loses a worker. How long until it trains
again, how much work does it lose, and what does the protection cost?
This repo measures it on [Modal](https://modal.com) with DDP and FSDP2,
DCP checkpoints, and injected faults: a crashed rank, a hung rank, and a
lost container.

## Setup

```sh
uv sync
uv run modal setup   # once
uv run mlib run data # the FineWeb-edu token shards, to the Volume faults-data
```

`modal.json` has the recipes, and `train.py` documents its settings.
[modal-lib](https://github.com/crsmithdev/modal-lib) runs them and records
the time and cost of each run.
