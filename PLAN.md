# Cars n' Faces: Step-by-Step Plan

The idea, decisions and stack are in [`CONTEXT.md`](CONTEXT.md). This file is the order of work. Tick boxes as you go.

**Stack:** Colab Pro (GPU) · PyTorch (from scratch) · Weights & Biases free plan · Google Drive (data + checkpoints) · Claude artifact (survey)
**Datasets (no approval needed, for now):** CelebA, Google Cartoon Set, Stanford Cars

## Project layout

```
carsnfaces/
├── CONTEXT.md                  idea, decisions, open questions
├── PLAN.md                     this file
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
│   ├── download_data.sh        CelebA / Cartoon Set / Stanford Cars
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

- [ ] 1.1 Download CelebA and Stanford Cars (notebook 01, "Download data").
- [ ] 1.2 Download Cartoon Set (needs Google sign-in in Colab; no approval).
- [ ] 1.3 **Pick car fronts by hand:** browse Stanford Cars and list ~60 clear, roughly frontal views in `data/selections/car_fronts.txt` (paths relative to the dataset folder, e.g. `cars_train/00042.jpg`). Aim for variety: sedans, SUVs, trucks, sports cars, old/new, "angry" and "friendly" looking ones. Also list ~10 rear views in `car_rears.txt` (controls).
  - Stanford Cars has few pure front views; tight 3/4 views are acceptable if both headlights are visible.
  - *Optional helper:* a small picker page (artifact) to click through images and export the list.
- [ ] 1.4 Run `prepare_images.py` for fronts and rears at 64px. Look at the result: the car should fill most of the square.

**Done when:** `data/processed/car_fronts_64/` has ~60 good images.

## Phase 2: Trait pilot (in parallel with phase 3, ~1 day of waiting)

- [ ] 2.1 Show ~20 of the selected cars to 3–5 people. Ask: *"If this car were a face, describe its personality/expression in a few words."*
- [ ] 2.2 Count the most common words → pick the **final 4 traits** (draft: angry, happy, sad, chubby). Record in `CONTEXT.md`.
- [ ] 2.3 Drop cars nobody saw a face in, or note them as a "low face-likeness" subset.

**Done when:** 4 traits are fixed and the 40 study cars are chosen.

## Phase 3: Main model, real faces, 64px (~1–2 days)

- [ ] 3.1 Smoke test: `configs/smoke_test.yaml` (runs in seconds, no W&B).
- [ ] 3.2 Train `configs/faces_64.yaml` (30k steps; expect ~1–2 h on a Colab GPU). Watch in W&B:
  - `train/loss` and `val/loss` going down
  - `samples/faces`: held-out faces reconstructed recognizably
  - `samples/cars_to_faces`: the first car→face results
- [ ] 3.3 Run `notebooks/02_cars_to_faces.ipynb` on fronts **and** rears. Look at the pairs.
- [ ] 3.4 **Sanity checks** (these decide whether the idea works):
  - **Template check:** shift/scale a few cars inside the frame. Do the eyes follow the headlights, or stay at fixed pixels? If fixed, increase augmentation.
  - **Selectivity check:** do fronts give clearer faces than rears?
  - **Variety check:** do different cars give *different* faces, or always the same average face?
- [ ] 3.5 Write down findings in `CONTEXT.md` (decisions section).

**Done when:** car→face outputs are varied and follow the car's structure. If faces are all the same blur, go to phase 4 before anything else.

## Phase 4: Improve the model (iterate as needed)

Try one change at a time, each as a new config + W&B run, compared on the same probe cars:

- [ ] 4.1 Latent size (64 / 128 / 256 / 512): a smaller latent gives more "face-like" outputs, a bigger one copies the car more. This is the key knob.
- [ ] 4.2 Color vs grayscale.
- [ ] 4.3 Sharper outputs: VAE, or adversarial loss (discriminator trained from scratch), still no pretrained networks.
- [ ] 4.4 Scale up to **128px** once 64px works (new config `faces_128.yaml`).

**Done when:** one config is chosen as *the* main model; its checkpoint is frozen.

## Phase 5: Secondary model, Cartoon Set

- [ ] 5.1 Train `configs/cartoon_64.yaml` with the architecture chosen in phase 4.
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

- CompCars (viewpoint labels, but needs a request form) for more and better front views.
- Other pareidolia objects (Faces in Things dataset).
- Diffusion / SDEdit comparison.
