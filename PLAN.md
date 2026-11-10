# Cars n' Faces: Step-by-Step Plan

The idea, decisions and stack are in [`CONTEXT.md`](CONTEXT.md). This file is the order of work. Tick boxes as you go.

**Stack:** Colab Pro (GPU) · PyTorch (from scratch) · Weights & Biases free plan · Google Drive (data + checkpoints) · Claude artifact (survey)
**Datasets (no approval needed, for now):** CelebA, Google Cartoon Set, DVM-CAR

## Project layout

```
carsnfaces/
├── CONTEXT.md                  idea, decisions, open questions
├── PLAN.md                     this file
├── BASELINE.md                 spec of the first training run
├── configs/                    one YAML per experiment
│   ├── faces_64.yaml           main model (CelebA)
│   ├── cartoon_64.yaml         secondary model (Cartoon Set)
│   └── smoke_test.yaml         tiny pipeline check
├── src/carsnfaces/
│   ├── config.py               YAML → Config; ${CNF_DATA}/${CNF_RUNS} paths
│   ├── data.py                 image folder dataset + augmentation
│   ├── model.py                conv autoencoder
│   ├── train.py                training loop, W&B logging, auto-resume
│   └── infer.py                cars → faces (+ latents)
├── scripts/
│   ├── download_data.sh        CelebA / Cartoon Set / DVM-CAR
│   └── prepare_images.py       select + square + resize images
├── notebooks/
│   ├── 01_train_autoencoder.ipynb
│   └── 02_cars_to_faces.ipynb
├── data/selections/            hand-picked car lists (tracked in git)
├── survey/                     Claude artifact survey (phase 6)
├── analysis/                   stats on survey results (phase 7)
└── data/ runs/ outputs/        local only, gitignored (on Colab these live in Drive)
```

## Phase 0: Setup (~1 hour)

- [ ] 0.1 Create a free **W&B** account and copy the API key (https://wandb.ai/authorize).
- [ ] 0.2 Create a free **Kaggle** account → Settings → API → *Create New Token* → `kaggle.json`.
- [ ] 0.3 Copy this folder to **Google Drive** at `MyDrive/carsnfaces` (or sync it with Google Drive for desktop). Put `kaggle.json` in `MyDrive/carsnfaces/secrets/`.
- [ ] 0.4 *(Optional)* `git init` + private GitHub repo for version history. Data, checkpoints and `secrets/` are already gitignored.
- [ ] 0.5 Open `notebooks/01_train_autoencoder.ipynb` in Colab (GPU runtime) and run the setup + credentials cells.

**Done when:** the notebook mounts Drive, installs the package and logs in to W&B.

## Phase 1: Data (~half a day)

Details in [`BASELINE.md`](BASELINE.md) §1.

- [ ] 1.1 Download CelebA; preprocess to 80×80 grayscale NumPy arrays using the official train/val/test split.
- [ ] 1.2 Download DVM-CAR quality-checked front views (730 MB) and the tables (CSV).
- [ ] 1.3 Download the full DVM-CAR image set to Colab's local disk; use `Image_table.csv` to extract a few hundred **rear** views; save only those to Drive.
- [ ] 1.4 Preprocess cars: crop to the car → pad to square (white) → 64×64 grayscale. Look at the result: the car should fill the frame.
- [ ] 1.5 Save the fixed probe/eval lists (`probe_faces`, `probe_fronts`, `probe_rears`, `eval_*`) to `data/selections/`.
- [ ] 1.6 Download Cartoon Set (needs Google sign-in in Colab; no approval). Only needed for phase 5.
- [ ] 1.7 **Pick the ~60 study cars by hand** from the DVM fronts (40 for the survey + spares) into `data/selections/car_fronts.txt`. Aim for variety: sedans, SUVs, vans, sports cars, old/new, "angry" and "friendly" looking ones.
  - *Optional helper:* a small picker page (artifact) to click through images and export the list.

**Done when:** CelebA arrays, preprocessed fronts/rears and the probe lists are on Drive.

## Phase 2: Trait pilot (in parallel with phase 3, ~1 day of waiting)

- [ ] 2.1 Show ~20 of the selected cars to 3–5 people. Ask: *"If this car were a face, describe its personality/expression in a few words."*
- [ ] 2.2 Count the most common words → pick the **final 4 traits** (draft: angry, happy, sad, chubby). Record in `CONTEXT.md`.
- [ ] 2.3 Drop cars nobody saw a face in, or note them as a "low face-likeness" subset.

**Done when:** 4 traits are fixed and the 40 study cars are chosen.

## Phase 3: Main model, real faces, 64px: β-VAE sweep over β and latent size (~2–4 days)

The first model is a **β-VAE** (a variational autoencoder whose loss is `reconstruction + β · KL`). Two knobs control how strongly a car is pushed toward a face:

| Knob | Low value | High value |
|---|---|---|
| **β** (weight of the KL term) | Copies the input closely, so a car comes out looking like a car | Pulled toward the "average face", so every car may give the same generic face |
| **Latent size** | Squeezes everything into a few face features, giving more face-like but less car-specific outputs | Keeps more detail, so the car leaks through |

The interesting zone is in the middle: outputs that are clearly **faces** but still **different for each car**. The sweep maps that trade-off.

**Code needed first** (the scaffold currently has a plain autoencoder): add a VAE variant to `model.py` (encoder outputs μ and log σ², reparameterization), `beta` and `kl_warmup_steps` config fields, and log `train/recon`, `train/kl` and random-prior samples to W&B.

- [ ] 3.1 Smoke test: `configs/smoke_test.yaml` (runs in seconds, no W&B).
- [ ] 3.2 **Baseline run:** residual β-VAE, β = 1, latent = 128, 100 epochs. Spec and step-by-step run plan in [`BASELINE.md`](BASELINE.md). Its learning curve sets the sweep run length.
- [ ] 3.3 **Coarse sweep** (W&B Sweeps, free plan; run length set by the baseline's learning curve, shorter than 100 epochs):
  - β ∈ {0.1, 0.5, 1, 2, 4, 8}
  - latent ∈ {16, 32, 64, 128, 256}
  - 30 runs. If Colab time is tight, start with the 3×3 corner-and-middle grid β {0.5, 2, 8} × latent {32, 128, 256}.
  - Use the same seed, data split, probe cars and KL warm-up in every run.
- [ ] 3.4 **Compare runs** on the same probe cars (fronts and rears). For each run log:
  - **Face fidelity:** validation reconstruction loss on held-out faces.
  - **KL / active latent units:** how many latent dimensions are actually used.
  - **Face-likeness of car outputs:** how close the car's latent is to the face distribution (e.g. KL of the car's posterior to the prior), and optionally a classic, non-deep face detector (OpenCV Haar cascade) on the outputs.
  - **Car specificity:** diversity of outputs across cars (mean pairwise distance), and whether similar cars give similar faces (correlation between car–car and face–face distances).
  - **Front vs rear gap:** fronts should be more face-like than rears.
  - **Visual grid:** the same 16 cars through every run, side by side (W&B panel or a notebook figure).
- [ ] 3.5 **Plot the trade-off:** face-likeness vs car specificity, one point per (β, latent) run. Pick 1–3 configs on the "good" edge and run a **fine sweep** around them (e.g. β ±50%, neighboring latent sizes) at full length.
- [ ] 3.6 **Sanity checks** on the chosen config(s):
  - **Template check:** shift/scale a few cars inside the frame. Do the eyes follow the headlights, or stay at fixed pixels? If fixed, increase augmentation.
  - **Selectivity check:** do fronts give clearer faces than rears?
  - **Variety check:** do different cars give *different* faces?
- [ ] 3.7 Record the sweep results and the chosen (β, latent) in `CONTEXT.md`. The β/latent trade-off plot is itself a result for the paper.

**Done when:** one (β, latent) config is chosen whose car→face outputs are clearly faces, vary between cars, and follow the car's structure.

## Phase 4: Improve the model (iterate as needed)

Try one change at a time on the chosen (β, latent), each as a new config + W&B run, compared on the same probe cars:

- [ ] 4.1 Color vs grayscale.
- [ ] 4.2 Sharper outputs: adversarial loss (VAE-GAN, discriminator trained from scratch), still no pretrained networks.
- [ ] 4.3 Scale up to **128px** once 64px works (new config `faces_128.yaml`). Re-check β, since the best value can shift with resolution.
- [ ] 4.4 Ideas from [`BASELINE.md`](BASELINE.md) §9: attention, other downsampling, Gaussian decoder with learned σ, β-TCVAE.

**Done when:** one config is chosen as *the* main model; its checkpoint is frozen.

## Phase 5: Secondary model, Cartoon Set

- [ ] 5.1 Train `configs/cartoon_64.yaml` with the architecture and (β, latent) chosen in phases 3–4 (optionally a small β sweep, since cartoons are simpler and may need a smaller latent).
- [ ] 5.2 Generate faces for the ~20 shared cars; run the same sanity checks.

**Done when:** cartoon faces exist for the shared 20 cars.

## Phase 6: Survey (Claude artifact)

- [ ] 6.1 Export stimuli: 40 cars, 40 realistic faces, 20 cartoon faces, plus a `manifest.json` with matching pairs and distractors (distractor = face from another car).
- [ ] 6.2 Build the survey page (see `survey/README.md`): consent → random Group A/B → 4-trait ratings → matching test. Responses go to the artifact database.
- [ ] 6.3 Pilot with 1–2 people: opens fine, works on phones, ~15 min, responses saved.
- [ ] 6.4 Send to **15–30 people**. Track Group A/B balance.

**Done when:** ≥15 complete responses, groups roughly balanced.

## Phase 7: Analysis

- [ ] 7.1 Export responses; clean (drop incomplete / too-fast responses).
- [ ] 7.2 Rater reliability (split-half).
- [ ] 7.3 **Main result:** per-trait correlation between car ratings (Group A) and face ratings (Group B).
- [ ] 7.4 Matching accuracy vs 50% chance.
- [ ] 7.5 Exploratory: realistic vs cartoon.
- [ ] 7.6 Figures: car | face pairs for the best and worst matches; scatter plots per trait.

## Phase 8: Write-up

- [ ] 8.1 Short paper: intro (pareidolia, related work), method, model results, human study, discussion.
- [ ] 8.2 Optional: a public results page (artifact) with interactive car→face examples.

## Later (not now)

- CompCars (needs a request form) or Wikimedia Commons front views, for more varied real-world car photos.
- Other pareidolia objects (Faces in Things dataset).
- Diffusion / SDEdit comparison.
