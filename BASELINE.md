# Baseline Run: Spec & Run Plan

The first real training run (PLAN.md step 3.2). Every later run in the β / latent sweep changes **only** β and the latent size (and the number of steps); everything else here stays fixed so runs are comparable.

**One sentence:** a 64×64 grayscale **residual convolutional β-VAE** (β = 1, latent = 128) trained from scratch on CelebA for **100 epochs**, evaluated on fixed sets of CelebA faces and DVM-CAR car fronts and rears.

## 1. Data

### Faces (training): CelebA

| | |
|---|---|
| Source | CelebA "aligned & cropped" (Kaggle `jessicali9530/celeba-dataset`), 202,599 images, 178×218 |
| Splits | **Official CelebA partition** (`list_eval_partition.csv`): train 162,770 / val 19,867 / test 19,962. Test stays untouched until the final model. |
| Crop | Center crop **148×148**: the whole head including hair and chin, because a car front is a whole "head" with an outline, not just inner features |
| Stored as | Grayscale, resized to **80×80**, one `uint8` NumPy array per split (train ≈ 1.0 GB). Saved to Drive once, copied to Colab's local disk each session, and loaded into RAM. This avoids reading 160k small files from Drive, which is very slow. |
| Augmentation (train only, on GPU) | Random scale (area 70–100%) + random position → resized to **64×64**, rotation ±10°, horizontal flip. This is what stops the model from always putting eyes at the same pixels. |
| Eval inputs | Center 64×64 view (resize 80→64), no augmentation |

### Cars (evaluation only, never trained on): DVM-CAR

| | |
|---|---|
| Source | [DVM-CAR](https://deepvisualmarketing.github.io/) 2.0, CC BY-NC 4.0, direct download, no approval |
| Fronts | "Quality checked front-view images" set: **61,827** images, 730 MB |
| Rears (control) | From the full image set (13.6 GB, downloaded to Colab's local disk only), filtered with `Image_table.csv` → `Predicted_viewpoint` (front = 0; rear code to be checked in the table). Keep a few hundred rears; the full set is not stored. |
| Format | 300×300 JPG, **background already removed (white)**, filenames encode brand / model / year / colour |
| Preprocessing | Grayscale → crop to the car's bounding box (non-white pixels) + 5% margin → pad to square with white → resize to **64×64**. This makes each car fill the frame, roughly as a head fills a CelebA crop. Same pipeline for fronts and rears. |

**Why DVM-CAR over Stanford Cars:** front views are already selected and checked by hand, there's no manual filtering of a mixed-angle dataset, it has same-style rear views for controls, and the clean background removes clutter.

### Fixed probe sets (saved as file lists in `data/selections/`, same for every run)

| Set | Size | Use |
|---|---|---|
| `probe_faces` | 16 CelebA val images | Reconstruction quality |
| `probe_fronts` | 32 DVM fronts, random with seed 0, varied brands | Car→face outputs |
| `probe_rears` | 32 DVM rears, same brands/models as the fronts where possible | Control: should be less face-like |
| `eval_faces` | 2,000 CelebA val images | Metrics |
| `eval_fronts` / `eval_rears` | 1,000 each | Metrics |

The ~40 **study** cars for the human survey are chosen later by hand from the DVM fronts.

## 2. Architecture: residual convolutional β-VAE

Input `1×64×64` in [0, 1]. Norm = **GroupNorm (32 groups)**, activation = **SiLU**. GroupNorm normalizes each image on its own, so a car is processed independently of the rest of its batch, and train and eval behave the same.

**Residual block** `ResBlock(c_in → c_out)`:

```
h = Conv3×3( SiLU( GN(x) ) )          # c_in → c_out
h = Conv3×3( SiLU( GN(h) ) )          # c_out → c_out
out = h + skip(x)                     # skip = identity, or Conv1×1 if c_in ≠ c_out
```

**Downsampling:** stride-2 Conv 4×4 (unchanged from the plain design). **Upsampling:** nearest ×2 → Conv 3×3.

**Encoder**

| Step | Output |
|---|---|
| Stem: Conv 3×3 | 64 × 64 × 64 |
| 2× ResBlock(64) → Down | 64 × 32 × 32 |
| 2× ResBlock(→128) → Down | 128 × 16 × 16 |
| 2× ResBlock(→256) → Down | 256 × 8 × 8 |
| 2× ResBlock(→512) → Down | 512 × 4 × 4 |
| ResBlock(512) → GN → SiLU → Flatten → Linear | **μ** (128), **log σ²** (128, clamped to [−10, 10]) |

**Latent:** `z = μ + σ · ε`, ε ~ N(0, I) during training. At eval, **decode μ** (no sampling), so a car always gives the same face.

**Decoder** (mirror of the encoder)

| Step | Output |
|---|---|
| Linear → reshape → ResBlock(512) | 512 × 4 × 4 |
| Up → 2× ResBlock(→512) | 512 × 8 × 8 |
| Up → 2× ResBlock(→256) | 256 × 16 × 16 |
| Up → 2× ResBlock(→128) | 128 × 32 × 32 |
| Up → 2× ResBlock(→64) | 64 × 64 × 64 |
| GN → SiLU → Conv 3×3 | 1 × 64 × 64 **logits** (sigmoid only for display) |

**Size:** roughly 40M parameters (the code will print the exact count). If it's too slow, cap channels at 256 (about 4× fewer parameters) and note it.

**Deliberately not included:**
- **Skip connections between encoder and decoder** (U-Net style). They would let car pixels bypass the latent and leak straight into the output.
- **Spatial latent:** the single 128-number vector is what forces a face.
- **Attention, other downsampling:** postponed (§9).

## 3. Loss

```
loss = recon + β_t · KL
recon = BCE-with-logits(x̂, x), summed over the 4,096 pixels, averaged over the batch
KL    = KL(q(z|x) ‖ N(0, I)), summed over latent dims, averaged over the batch
β_t   = β · min(1, step / 5,000)        # linear KL warm-up
```

- **β = 1** for the baseline.
- Summing (not averaging) over pixels makes β = 1 the standard VAE objective, so the sweep values (0.1 … 8) have a clear meaning. **Every run must use this same normalization.**
- Warm-up stops the KL term from collapsing the latent early in training.

## 4. Training

| | |
|---|---|
| Optimizer | Adam, lr **5e-4**, betas (0.9, 0.999), no weight decay |
| LR schedule | 1,000-step linear warm-up, then cosine decay to 5e-5 at the last step |
| Batch size | 128 (≈ 1,272 steps per epoch) |
| Length | **100 epochs ≈ 127,000 steps** |
| Precision | bf16 autocast on A100/L4, fp16 + GradScaler on T4; KL computed in fp32 |
| Gradient clipping | None at first; add (norm 100) only if loss spikes or NaNs appear |
| Seed | 0 (weights, data order, augmentation) |
| Checkpoints | `last.pt` every 5,000 steps to Drive (auto-resume) + **milestones kept permanently at epochs 5, 10, 25, 50, 100** |
| Time | Measured in the speed test (§7, step 3) |

**Why 100 epochs:** VAEs on CelebA keep improving slowly for a long time. The long run gives the full learning curve, and the milestone checkpoints show whether car→face outputs change as the encoder gets more precise (does the car start leaking into the face?). The learning curve also decides how short the sweep runs can be.

## 5. Logging (W&B)

| Every | What |
|---|---|
| 100 steps | `train/loss`, `train/recon`, `train/kl`, `train/beta_t`, `lr`, images/s |
| 2,500 steps | On `eval_*`: val recon and KL; **active units** (latent dims with variance of μ across faces > 0.01); mean KL of fronts vs rears vs faces; diversity of car outputs (mean pairwise distance) |
| 2,500 steps | Image grids: probe faces (input/recon), probe fronts → faces, probe rears → faces, 64 random samples from the prior |
| Milestones | Same grids saved to Drive as PNGs (for the "over training" figure) |
| End | All metrics as a W&B summary; `final.pt` as a W&B artifact `baseline-v1` |

## 6. Success criteria

| Check | Pass if |
|---|---|
| Trains | Loss falls smoothly, no NaNs |
| Faces reconstruct | Probe faces recognizable (blur is fine at this stage) |
| Prior samples | Random latents decode to plausible faces |
| No collapse | KL clearly > 0; at least ~20 of 128 active units |
| Cars → faces | Fronts produce **faces** (not cars, not gray blobs) |
| Variety | Different fronts give visibly different faces |
| Front vs rear | Fronts look more face-like than rears (by eye; the metric is a first hint) |

## 7. Run plan (step by step)

**Before:** PLAN.md phase 1 done (CelebA arrays, DVM fronts/rears, probe lists on Drive) and the code changes in §10 written.

1. **Smoke test** (Colab, any GPU, ~5 min): `smoke_test.yaml`, 200 steps, tiny model, W&B off.
   - Check: tensor shapes, loss goes down, grids save, a checkpoint is written.
   - **Resume test:** stop the cell mid-run, re-run it, and confirm it continues from the last step.
2. **Short W&B test** (~10 min): full model, 2,000 steps, W&B on.
   - Check: every metric and grid from §5 appears; car grids show the right images; the active-units and KL numbers look sane.
3. **Speed test**: 1,000 steps on an **L4**, and on an A100 if available. Note images/s → time for 127k steps → compute units for each GPU. Pick the GPU (L4 is likely the best value at 64px). If the full run would take more than ~8 h, cap channels at 256 and re-test.
4. **Launch the full run** (`faces_64_baseline.yaml`). Check in at:
   - **~2k steps:** loss falling, face reconstructions starting to form.
   - **~10k steps (after KL warm-up):** KL stable, active units > 20, no NaNs.
   - **Epochs 25 and 50:** val loss still falling? Car grids changing?
   - If a session drops, re-run the training cell; it resumes from `last.pt`.
5. **Inference:** run `02_cars_to_faces` with `final.pt` **and** each milestone checkpoint on `probe_fronts`, `probe_rears` and the eval sets.
6. **Figures:**
   - Probe fronts → faces at epochs 5 / 10 / 25 / 50 / 100 side by side.
   - Learning curves: val recon, KL, active units, front-vs-rear KL gap.
7. **Evaluate** against §6 and fill in §8.
8. **Set the sweep length:** find the epoch after which the car→face metrics (diversity, front-vs-rear gap) and the face reconstruction stop changing much. That becomes the step count for the coarse sweep (PLAN.md 3.3).
9. **Record:** results in §8, a summary + decision in `CONTEXT.md`, W&B artifact `baseline-v1`, a git commit tagged `baseline-v1`.

## 8. Results (fill in after the run)

| | |
|---|---|
| Date / GPU / wall time / compute units | |
| Parameters | |
| Final val recon / KL | |
| Active units (of 128) | |
| Front vs rear KL gap | |
| Car output diversity | |
| Success criteria passed | |
| Observations (car grids over training) | |
| Chosen sweep length | |
| W&B run link | |

## 9. Future (not in the baseline)

Ideas to test later, one at a time against the baseline, on the same probe cars.

**Architecture**
- **Attention** (self-attention block at 8×8 in encoder and decoder): links distant features (left "eye" ↔ right "eye"), which may give a more coherent face layout.
- **Other downsampling:** average pooling + conv, or blur-pool, instead of stride-2 convs.

**Loss**
- **Gaussian decoder with a learned σ (σ-VAE):** replace BCE with a Gaussian likelihood whose noise level σ is learned. It balances reconstruction vs KL automatically and is often sharper than BCE. Note: changes what β = 1 means, so the sweep must be redone or re-calibrated.
- **β-TCVAE:** split the KL into mutual information, total correlation and dimension-wise KL, and weight the total-correlation term. It encourages each latent dimension to capture one factor (smile, face width, …), which may expose **trait directions** (e.g. "chubby", "angry") to relate to the survey traits.
- Also noted earlier: SSIM / frequency losses, perceptual loss from a self-trained face network, adversarial loss (PLAN.md phase 4).

## 10. Code changes needed before running

The scaffold has a plain autoencoder and an untested pipeline. Before this run:

- `model.py`: residual β-VAE as in §2 (ResBlock, GroupNorm, SiLU, μ / log σ², logits output).
- `train.py`: β-VAE loss, KL warm-up, warm-up + cosine LR, step count from epochs, milestone checkpoints, §5 metrics and grids, official CelebA split, NumPy in-RAM dataset, GPU augmentation.
- `config.py` + new `configs/faces_64_baseline.yaml`: `beta`, `kl_warmup_steps`, `lr_warmup_steps`, `lr_min`, `epochs`, `milestone_epochs`, `channels`, `res_blocks_per_stage`.
- `scripts/`: CelebA → 80px NumPy preprocessing; DVM-CAR download + car crop/pad; rear filtering from `Image_table.csv`; probe-set lists.
- Drop Stanford Cars from the download script.
