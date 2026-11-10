"""Train the face beta-VAE (BASELINE.md).

    python -m carsnfaces.train --config configs/faces_64_baseline.yaml
    python -m carsnfaces.train --config configs/smoke_test.yaml --set max_steps=50

Resumes automatically from ``<run_dir>/last.pt`` (Colab sessions disconnect).
Data order and augmentation are seeded per epoch / per step, so a resumed run
sees the same batches it would have seen without the interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image

from .config import Config, load_config
from .data import augment, center_view, load_array, read_names, select, to_float
from .model import build_model, kl_per_sample, recon_per_sample


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def pick_precision(cfg: Config, device: str):
    """-> (autocast dtype or None, use GradScaler)."""
    p = cfg.precision
    if device != "cuda" or p == "fp32":
        return None, False
    if p == "auto":
        p = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return (torch.bfloat16, False) if p == "bf16" else (torch.float16, True)


def lr_lambda(cfg: Config, total: int):
    floor = cfg.lr_min / cfg.lr

    def f(step: int) -> float:
        if step < cfg.lr_warmup_steps:
            return (step + 1) / cfg.lr_warmup_steps
        t = (step - cfg.lr_warmup_steps) / max(1, total - cfg.lr_warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


class EvalData:
    """Fixed probe / eval sets (BASELINE.md §1), held on the device as float images."""

    def __init__(self, cfg: Config, device: str):
        d = Path(cfg.data_dir)
        sel = d / "selections"
        faces_val = load_array(d / f"{cfg.faces}_val.npy")
        face_names = read_names(d / f"{cfg.faces}_val.txt")
        fronts, front_names = load_array(d / f"{cfg.fronts}.npy", mmap=True), read_names(d / f"{cfg.fronts}.txt")
        rears, rear_names = load_array(d / f"{cfg.rears}.npy", mmap=True), read_names(d / f"{cfg.rears}.txt")

        def take(arr, names, listfile, size):
            idx = select(names, read_names(sel / listfile))
            x = torch.from_numpy(np.ascontiguousarray(arr[idx]))
            return center_view(x.to(device), size)

        s = cfg.image_size
        self.probe_faces = take(faces_val, face_names, "probe_faces.txt", s)
        self.eval_faces = take(faces_val, face_names, "eval_faces.txt", s)
        self.probe_fronts = take(fronts, front_names, "probe_fronts.txt", s)
        self.eval_fronts = take(fronts, front_names, "eval_fronts.txt", s)
        self.probe_rears = take(rears, rear_names, "probe_rears.txt", s)
        self.eval_rears = take(rears, rear_names, "eval_rears.txt", s)


def interleave(top: torch.Tensor, bottom: torch.Tensor, per_row: int) -> torch.Tensor:
    """Rows of `per_row` inputs, each followed by a row of their outputs."""
    rows = []
    for i in range(0, top.shape[0], per_row):
        rows += [top[i:i + per_row], bottom[i:i + per_row]]
    return make_grid(torch.cat(rows), nrow=per_row, padding=2, pad_value=1.0)


@torch.no_grad()
def evaluate(model, ev: EvalData, cfg: Config, step: int, amp_dtype) -> tuple[dict, dict]:
    """-> (scalar metrics, image grids as tensors)."""
    model.eval()
    dev = ev.eval_faces.device

    def run(x, bs=500):
        mus, lvs, ys, rec = [], [], [], []
        for i in range(0, x.shape[0], bs):
            xb = x[i:i + bs]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                mu, lv = model.encode(xb)
                logits = model.decode(mu)
            mus.append(mu.float()); lvs.append(lv.float())
            ys.append(torch.sigmoid(logits.float())); rec.append(recon_per_sample(logits, xb))
        return torch.cat(mus), torch.cat(lvs), torch.cat(ys), torch.cat(rec)

    def diversity(y):
        f = y.flatten(1)
        d = torch.cdist(f, f) / math.sqrt(f.shape[1])     # RMS pixel distance between outputs
        n = f.shape[0]
        return (d.sum() / (n * (n - 1))).item()

    m = {}
    mu_f, lv_f, _, rec_f = run(ev.eval_faces)
    kl_f = kl_per_sample(mu_f, lv_f)
    m["val/recon"] = rec_f.mean().item()
    m["val/kl"] = kl_f.mean().item()
    m["val/loss_beta1"] = m["val/recon"] + m["val/kl"]
    m["val/active_units"] = int((mu_f.var(dim=0) > 0.01).sum().item())

    for name, x in (("fronts", ev.eval_fronts), ("rears", ev.eval_rears)):
        mu, lv, y, _ = run(x)
        m[f"cars/kl_{name}"] = kl_per_sample(mu, lv).mean().item()
        m[f"cars/diversity_{name}"] = diversity(y)
    m["cars/kl_faces"] = m["val/kl"]
    m["cars/kl_gap_rear_minus_front"] = m["cars/kl_rears"] - m["cars/kl_fronts"]

    g = torch.Generator(device=dev).manual_seed(1234)   # same prior samples every time
    z = torch.randn(64, cfg.latent_dim, device=dev, generator=g)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        prior = torch.sigmoid(model.decode(z).float())

    grids = {
        "faces": interleave(ev.probe_faces, model.reconstruct(ev.probe_faces), 16),
        "fronts": interleave(ev.probe_fronts, model.reconstruct(ev.probe_fronts), 16),
        "rears": interleave(ev.probe_rears, model.reconstruct(ev.probe_rears), 16),
        "prior": make_grid(prior, nrow=8, padding=2, pad_value=1.0),
    }
    model.train()
    return m, grids


def save_checkpoint(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a disconnect mid-save won't corrupt last.pt


def train(cfg: Config) -> None:
    import wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    run_dir = Path(cfg.run_dir)
    (run_dir / "milestones").mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ---- data: whole training set on the GPU as uint8 (~1 GB for CelebA at 80px)
    t = time.time()
    train_np = load_array(Path(cfg.data_dir) / f"{cfg.faces}_train.npy", mmap=True)
    if cfg.max_train_images:
        train_np = train_np[: cfg.max_train_images]
    train_x = torch.from_numpy(np.ascontiguousarray(train_np)).to(device)
    n_train = train_x.shape[0]
    ev = EvalData(cfg, device)
    print(f"[data] {n_train} train faces {tuple(train_x.shape[1:])}, eval sets loaded in {time.time() - t:.0f}s")

    steps_per_epoch = n_train // cfg.batch_size
    total = cfg.max_steps or int(round(cfg.epochs * steps_per_epoch))
    milestones = {e * steps_per_epoch: e for e in cfg.milestone_epochs if e * steps_per_epoch <= total}

    # ---- model / optimizer
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    amp_dtype, use_scaler = pick_precision(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step, wandb_id = 0, None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step, wandb_id = ck["step"], ck.get("wandb_id")
        print(f"[resume] from step {step}")
    encode, decode = (torch.compile(model.encode), torch.compile(model.decode)) if cfg.compile \
        else (model.encode, model.decode)

    wandb_dir = os.environ.get("WANDB_DIR", str(run_dir))
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=cfg.wandb_project, name=cfg.name, config={**cfg.to_dict(), "params": n_params},
                     tags=cfg.tags, mode=cfg.wandb_mode, id=wandb_id, resume="allow", dir=wandb_dir)
    if cfg.wandb_mode != "disabled":
        wandb_id = run.id

    print(f"[train] {n_params / 1e6:.1f}M params on {device} "
          f"({torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}), precision {amp_dtype}, "
          f"{steps_per_epoch} steps/epoch, {total} steps total ({total / steps_per_epoch:.1f} epochs)")

    def state():
        return dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), step=step, config=cfg.to_dict(), wandb_id=wandb_id)

    def do_eval(save_to: Path | None = None):
        m, grids = evaluate(model, ev, cfg, step, amp_dtype)
        print("[eval] step {}: ".format(step) + "  ".join(
            f"{k.split('/')[-1]} {v:.3f}" if isinstance(v, float) else f"{k.split('/')[-1]} {v}" for k, v in m.items()))
        captions = {"faces": "rows: input / recon", "fronts": "rows: car front / face",
                    "rears": "rows: car rear / face", "prior": "random z ~ N(0, I)"}
        wandb.log({**m, **{f"img/{k}": wandb.Image(v.cpu(), caption=captions[k]) for k, v in grids.items()}},
                  step=step)
        if save_to is not None:
            save_to.mkdir(parents=True, exist_ok=True)
            for k, v in grids.items():
                save_image(v, save_to / f"{k}.png")
            (save_to / "metrics.json").write_text(json.dumps(m, indent=2))
        return m

    # ---- loop
    model.train()
    t0, seen = time.time(), 0
    perm_epoch, perm = -1, None
    while step < total:
        epoch, pos = divmod(step, steps_per_epoch)
        if epoch != perm_epoch:
            g = torch.Generator(device=device).manual_seed(cfg.seed * 100_003 + epoch)
            perm = torch.randperm(n_train, device=device, generator=g)
            perm_epoch = epoch
        idx = perm[pos * cfg.batch_size:(pos + 1) * cfg.batch_size]
        gen = torch.Generator(device=device).manual_seed(cfg.seed * 1_000_003 + step)
        x = augment(train_x[idx], cfg.image_size, cfg.aug_area_min, cfg.aug_rot_deg, cfg.aug_flip, gen)

        beta_t = cfg.beta * min(1.0, step / cfg.kl_warmup_steps) if cfg.kl_warmup_steps else cfg.beta
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            mu, logvar = encode(x)
            eps = torch.randn(mu.shape, device=device, generator=gen, dtype=mu.dtype)
            z = mu + eps * (0.5 * logvar).exp()
            logits = decode(z)
        recon = recon_per_sample(logits, x).mean()
        kl = kl_per_sample(mu, logvar).mean()
        loss = recon + beta_t * kl
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: recon {recon.item()} kl {kl.item()}")

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        seen += x.shape[0]

        if step % cfg.log_every == 0 or step == total:
            ips = seen / (time.time() - t0)
            lr = sched.get_last_lr()[0]
            wandb.log({"train/loss": loss.item(), "train/recon": recon.item(), "train/kl": kl.item(),
                       "train/beta_t": beta_t, "lr": lr, "img_per_s": ips, "epoch": step / steps_per_epoch},
                      step=step)
            eta_h = (total - step) * cfg.batch_size / ips / 3600
            print(f"step {step:>7}/{total}  ep {step / steps_per_epoch:6.2f}  loss {loss.item():8.1f}  "
                  f"recon {recon.item():8.1f}  kl {kl.item():6.1f}  beta_t {beta_t:.2f}  lr {lr:.2e}  "
                  f"{ips:5.0f} img/s  eta {eta_h:.1f}h", flush=True)
            t0, seen = time.time(), 0
        if step in milestones:
            e = milestones[step]
            do_eval(run_dir / "milestones" / f"epoch{e:03d}")
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "epoch": e},
                       run_dir / f"ckpt_epoch{e:03d}.pt")
        elif step % cfg.eval_every == 0:
            do_eval()
        if step % cfg.ckpt_every == 0:
            save_checkpoint(ckpt_path, **state())

    # ---- end
    m = do_eval(run_dir / "final")
    save_checkpoint(ckpt_path, **state())
    final = run_dir / "final.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, final)
    for k, v in m.items():
        run.summary[k] = v
    run.summary["params"] = n_params
    if cfg.wandb_artifact and cfg.wandb_mode != "disabled":
        art = wandb.Artifact(cfg.wandb_artifact, type="model", metadata={**m, "step": step})
        art.add_file(str(final))
        run.log_artifact(art)
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
