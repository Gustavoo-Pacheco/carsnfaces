"""Experiment config: a YAML file mapped onto a dataclass.

Paths may use environment variables, e.g. ``${CNF_DATA}/processed/celeba``.
Defaults: CNF_DATA=data, CNF_RUNS=runs (relative to the project root), so the
same config works locally and on Colab (where they point into Google Drive).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

import yaml

os.environ.setdefault("CNF_DATA", "data")
os.environ.setdefault("CNF_RUNS", "runs")


@dataclass
class Config:
    name: str = "faces_64"

    # data
    train_dir: str = "${CNF_DATA}/raw/celeba"
    probe_dir: str = "${CNF_DATA}/processed/car_fronts_64"  # cars shown in W&B during training
    image_size: int = 64
    grayscale: bool = True
    augment: bool = True  # random crop/shift/rotate so eyes aren't pinned to fixed pixels
    val_fraction: float = 0.02
    num_workers: int = 2

    # model
    base_channels: int = 64
    latent_dim: int = 256

    # training
    batch_size: int = 128
    lr: float = 2e-4
    loss: str = "l1"  # l1 | mse
    max_steps: int = 30_000
    seed: int = 0

    # logging / checkpoints
    log_every: int = 50
    sample_every: int = 1_000
    ckpt_every: int = 2_000
    run_dir: str = "${CNF_RUNS}/faces_64"
    wandb_project: str = "carsnfaces"
    wandb_mode: str = "online"  # online | offline | disabled

    tags: list[str] = field(default_factory=list)

    def resolved(self) -> "Config":
        """Return a copy with environment variables expanded in path fields."""
        data = asdict(self)
        for key in ("train_dir", "probe_dir", "run_dir"):
            data[key] = os.path.expandvars(data[key])
        return Config(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str, overrides: dict | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return Config(**raw).resolved()
