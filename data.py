"""Download the FineWeb-edu GPT-2 token shards that train.py reads.

The shards are in the llm.c format, 100M tokens each, from
karpathy/fineweb-edu-100B-gpt2-token-shards on Hugging Face. DATA is the
folder (data by default), and SHARDS is the count of training shards.
"""

import os
import pathlib
import urllib.request

URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main/"

data = pathlib.Path(os.environ.get("DATA", "data"))
data.mkdir(parents=True, exist_ok=True)
names = ["edu_fineweb_val_000000.bin"] + [
    f"edu_fineweb_train_{i:06d}.bin" for i in range(1, int(os.environ.get("SHARDS", "4")) + 1)]
for name in names:
    if (data / name).exists():
        continue
    urllib.request.urlretrieve(URL + name, data / f"{name}.tmp")
    os.replace(data / f"{name}.tmp", data / name)
    print(name, flush=True)
