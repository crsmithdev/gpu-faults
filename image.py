"""The image for the fault-tolerance recipes in modal.json."""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", "numpy==2.5.4", index_url="https://download.pytorch.org/whl/cu128",
                    extra_index_url="https://pypi.org/simple")
    .workdir("/root/faults")
    .add_local_dir(HERE, "/root/faults", ignore=["data", "results", "__pycache__"])
)
