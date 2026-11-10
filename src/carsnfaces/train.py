"""Train a face autoencoder.

    python -m carsnfaces.train --config configs/faces_64.yaml
    python -m carsnfaces.train --config configs/faces_64.yaml --set max_steps=200 wandb_mode=disabled

Resumes automatically from ``<run_dir>/last.pt`` (Colab sessions disconnect).
Logs loss, face reconstructions and car->face outputs to Weights & Biases.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .config import Config, load_config
from .data import ImageFolderFlat, build_transform, list_images
from .model import build_model


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def side_by_side(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Grid with each input directly above its reconstruction."""
    n = inputs.shape[0]
    return make_grid(torch.cat([inputs, outputs]), nrow=n, padding=2, pad_value=1.0)


def save_checkpoint(path: Path, model, opt, scaler, step: int, cfg: Config, wandb_id: str | None):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
        "wandb_id": wandb_id,
    }, tmp)
    tmp.replace(path)  # atomic: a disconnect mid-save won't corrupt last.pt


def train(cfg: Config) -> None:
    import wandb

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"

    # data: fixed train/val split by seed
    paths = list_images(cfg.train_dir)
    rng = random.Random(cfg.seed)
    rng.shuffle(paths)
    n_val = max(16, int(len(paths) * cfg.val_fraction))
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    train_ds = ImageFolderFlat(cfg.train_dir, build_transform(cfg.image_size, cfg.grayscale, cfg.augment),
                               paths=train_paths)
    eval_tf = build_transform(cfg.image_size, cfg.grayscale, augment=False)
    val_ds = ImageFolderFlat(cfg.train_dir, eval_tf, paths=val_paths)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=device == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    val_batch = torch.stack([val_ds[i][0] for i in range(min(16, len(val_ds)))]).to(device)

    probe_batch = None
    if Path(cfg.probe_dir).exists():
        probe_ds = ImageFolderFlat(cfg.probe_dir, eval_tf)
        probe_batch = torch.stack([probe_ds[i][0] for i in range(min(16, len(probe_ds)))]).to(device)
    else:
        print(f"[info] no probe cars at {cfg.probe_dir}; car->face samples won't be logged")

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = F.l1_loss if cfg.loss == "l1" else F.mse_loss

    step, wandb_id = 0, None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        step, wandb_id = ckpt["step"], ckpt.get("wandb_id")
        print(f"[resume] from step {step}")

    run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.to_dict(), tags=cfg.tags,
                     mode=cfg.wandb_mode, id=wandb_id, resume="allow", dir=str(run_dir))
    wandb_id = run.id if cfg.wandb_mode != "disabled" else None

    print(f"[train] {len(train_ds)} train / {len(val_ds)} val images on {device}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    @torch.no_grad()
    def log_samples():
        model.eval()
        payload = {"samples/faces": wandb.Image(side_by_side(val_batch, model(val_batch)).cpu(),
                                                caption="top: input, bottom: reconstruction")}
        payload["val/loss"] = loss_fn(model(val_batch), val_batch).item()
        if probe_batch is not None:
            payload["samples/cars_to_faces"] = wandb.Image(
                side_by_side(probe_batch, model(probe_batch)).cpu(), caption="top: car, bottom: face")
        wandb.log(payload, step=step)
        model.train()

    model.train()
    t0 = time.time()
    while step < cfg.max_steps:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % cfg.log_every == 0:
                ips = cfg.log_every * cfg.batch_size / (time.time() - t0)
                wandb.log({"train/loss": loss.item(), "train/img_per_s": ips}, step=step)
                print(f"step {step:>7}  loss {loss.item():.4f}  {ips:.0f} img/s")
                t0 = time.time()
            if step % cfg.sample_every == 0:
                log_samples()
            if step % cfg.ckpt_every == 0:
                save_checkpoint(ckpt_path, model, opt, scaler, step, cfg, wandb_id)
            if step >= cfg.max_steps:
                break

    log_samples()
    save_checkpoint(ckpt_path, model, opt, scaler, step, cfg, wandb_id)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, run_dir / "final.pt")
    print(f"[done] {step} steps, checkpoints in {run_dir}")
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()
    train(load_config(args.config, parse_overrides(args.set)))


if __name__ == "__main__":
    main()
