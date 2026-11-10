"""Experiment config: a YAML file mapped onto a dataclass.

Paths may use environment variables, e.g. ``${CNF_DATA}/processed``.
Defaults: CNF_DATA=data, CNF_RUNS=runs (relative to the project root), so the
same config works locally and on Colab (where they point into Google Drive).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

import yaml

os.environ.setdefault("CNF_DATA", "data")
os.environ.setdefault("CNF_RUNS", "runs")

PATH_FIELDS = ("data_dir", "run_dir")


@dataclass
class Config:
    name: str = "faces_64_baseline"

    # data (arrays written by scripts/prepare_celeba.py and scripts/prepare_dvm.py)
    data_dir: str = "${CNF_DATA}/processed"
    faces: str = "celeba80"          # <data_dir>/<faces>_{train,val}.npy, uint8 N x 80 x 80
    fronts: str = "dvm_fronts64"     # <data_dir>/<fronts>.npy + .txt (names), uint8 N x 64 x 64
    rears: str = "dvm_rears64"
    image_size: int = 64
    max_train_images: int = 0        # 0 = all (smoke test uses a subset)

    # augmentation (train faces only, on GPU)
    aug_area_min: float = 0.7        # random crop area 70-100% of the 80px image
    aug_rot_deg: float = 10.0
    aug_flip: bool = True

    # model: residual conv beta-VAE
    in_channels: int = 1
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    res_blocks_per_stage: int = 2
    gn_groups: int = 32
    latent_dim: int = 128
    logvar_clamp: float = 10.0

    # loss
    beta: float = 1.0
    kl_warmup_steps: int = 5_000

    # training
    batch_size: int = 128
    lr: float = 5e-4
    lr_min: float = 5e-5
    lr_warmup_steps: int = 1_000
    epochs: float = 100
    max_steps: int = 0               # >0 overrides epochs
    grad_clip: float = 0.0           # 0 = off
    precision: str = "auto"          # auto | bf16 | fp16 | fp32
    compile: bool = False
    seed: int = 0

    # logging / checkpoints
    log_every: int = 100
    eval_every: int = 2_500
    ckpt_every: int = 5_000
    milestone_epochs: list[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    run_dir: str = "${CNF_RUNS}/faces_64_baseline"
    wandb_project: str = "carsnfaces"
    wandb_mode: str = "online"       # online | offline | disabled
    wandb_artifact: str = ""         # e.g. baseline-v1: final.pt uploaded under this name
    tags: list[str] = field(default_factory=list)

    def resolved(self) -> "Config":
        """Return a copy with environment variables expanded in path fields."""
        data = asdict(self)
        for key in PATH_FIELDS:
            data[key] = os.path.expandvars(data[key])
        return Config(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_dict(raw: dict) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**raw)


def load_config(path: str, overrides: dict | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update(overrides or {})
    try:
        return config_from_dict(raw).resolved()
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
