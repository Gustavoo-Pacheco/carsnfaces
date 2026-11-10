# Cars n' Faces — Project Context

## One-line idea

Train a generative model **only on human faces**, then feed it **pictures of cars**. Because the model only knows how to produce faces, it is forced to "find" a face in each car. We then test whether the face the model finds is the **same face a human sees** in that car.

## Background

**Pareidolia** is the tendency to perceive meaningful patterns, especially faces, in objects that have none. Car fronts are a classic example: headlights read as eyes, the grille as a mouth, and people consistently describe some cars as "angry", "happy" or "cute". Neuroscience shows that illusory faces engage face-selective brain regions, so this is a real perceptual phenomenon, not just a figure of speech.

The open question: **can an artificial system reproduce human pareidolia, and does it see the same faces we do?**

## Core method

```
Training:  face ──► encoder ──► latent ──► decoder ──► face
Testing:   car  ──► encoder ──► latent ──► decoder ──► face (the face "seen" in the car)
```

1. Train an autoencoder (or similar generative model) on a large face dataset.
2. Pass car images through it. The output is always a face, the model's "interpretation" of the car.
3. Compare the model's interpretation with human perception of the same cars.

## Research questions

- **RQ1, alignment:** does the generated face match what humans see in the car (eyes on headlights, mouth on grille, same expression)?
- **RQ2, selectivity:** are car fronts more face-like to the model than car rears, other objects or noise?
- **RQ3, mechanism:** does the model actually read the car's structure, or does it impose a fixed face template? (Test by shifting, flipping and masking parts of the car.)
- **RQ4, (optional) generality:** does the effect hold across architectures (autoencoder, VAE, GAN inversion, diffusion)?

## Validation

**Automatic (cheap, run first):**
- Detect landmarks on generated faces and compare them with the car's headlights and grille.
- Run an expression classifier on generated faces.
- Use latent-space distance to real faces as a "face-likeness" score.

**Human (the core evidence):**
- **Matching test:** show a car and two generated faces (its own and another car's), and ask which belongs to the car. Above 50% means alignment.
- **Landmark clicks:** people mark where they see the eyes and mouth on each car.
- **Expression ratings:** people rate the emotion of each car and of each generated face, and we correlate the two.
- **Face-likeness ratings:** people rate how face-like each car is, and we correlate that with the model's score.

## Key design concerns

- **Trivial result:** a face-only model turns *anything* into a face, so controls (car rears, other objects, noise) are required.
- **Alignment bias:** face datasets are pre-aligned; without augmentation (shifts, scaling) the model may place eyes in fixed positions regardless of the input.
- **Blurriness:** plain autoencoders produce blurry faces, which humans can't judge; a perceptual or adversarial loss, or a pretrained generator, may be needed.
- **Color confound:** car color can leak into the face, so grayscale may help.

## Candidate resources

- **Faces:** CelebA, FFHQ
- **Cars:** CompCars (has viewpoint labels: front/rear), Stanford Cars
- **Related pareidolia data:** "Faces in Things" (Hamilton et al., 2024)
- **Compute:** Google Colab Pro
- **Related work to read:** Hamilton et al. 2024; Windhager et al. 2008; Kühn et al. 2014; Wardle et al. 2020/2022; Dobs et al. 2022

## Decisions (2026-11-10)

| Topic | Decision |
|---|---|
| Type of project | Personal project, solo |
| Participants | Has volunteers available to answer surveys |
| Models | **Everything built from scratch**, no pretrained generators |
| Face style | **Main model: realistic faces.** Secondary experiment: drawn faces |
| Scope | **Car fronts only** for now |
| Main claim | The model reproduces the **same car-to-trait relations humans perceive** (angry, sad, chubby, …) |
| Human tests | **Matching test** + **4 traits** |
| Participants | **15–30 people** |
| Language | English |
| Drawn-face dataset | **Google Cartoon Set** |
| Resolution | **64px first**, scale up (128px+) once the pipeline works |
| Survey | Custom website as a **Claude artifact** (responses in the artifact's built-in database) |
| Datasets | **Only datasets with no approval/request form, for now.** CelebA (Kaggle), Google Cartoon Set (direct download), **DVM-CAR** (Figshare/Dropbox, CC BY-NC 4.0: 61,827 checked front views, background removed; rears from the full set). Stanford Cars dropped; CompCars postponed. |
| Baseline run | 64px grayscale **residual** conv β-VAE, β = 1, latent 128, GroupNorm + SiLU, BCE summed + KL with warm-up, **100 epochs** with milestone checkpoints. Spec and run plan in `BASELINE.md`. |
| Future model ideas | Attention, other downsampling, Gaussian decoder with learned σ, β-TCVAE (`BASELINE.md` §9) |
| Stack | **Colab Pro + Weights & Biases (free plan)** |
| First model | **β-VAE** with a sweep over **β** and **latent size**, comparing how face-like vs car-specific the outputs are (PLAN.md phase 3) |

### Stack

| Layer | Tool | Use |
|---|---|---|
| Training compute | **Google Colab Pro** | GPU notebooks for training and inference |
| Framework | **PyTorch** | Models written from scratch (no pretrained generators) |
| Experiment tracking | **Weights & Biases (free plan)** | Loss curves, hyperparameters, reconstruction samples (faces *and* car→face outputs) logged every N steps, model checkpoints as artifacts |
| Storage | **Google Drive** (mounted in Colab) | Datasets, checkpoints, generated faces for the survey |
| Code | This folder (`~/Documents/InsperAI/carsnfaces`) | Notebooks and scripts, synced to Colab (via Drive or a GitHub repo) |
| Survey | **Claude artifact** | Trait ratings + matching test; responses stored in the artifact's database |

**Notes:**
- W&B free plan: keep logged media small (64px grids are cheap); store large checkpoints on Drive, only the best ones in W&B.
- Colab sessions disconnect, so training must save checkpoints to Drive and resume from the last one.

### Study design

**Main model:** an autoencoder trained from scratch on realistic faces (CelebA). All main results come from this model.

**Secondary experiment:** the same architecture trained on drawn faces. It goes through the same tests, but with fewer cars. Question: does a simpler, drawn face match human pareidolia better or worse than a realistic one?

**Human tests:**

1. **Matching test (2AFC):** a car is shown with two generated faces, its own and another car's: "Which face belongs to this car?" Above 50% correct means the model's face matches what people see.
2. **Trait ratings:** 4 traits, each on a 1–7 scale. Draft: **angry, happy, sad, chubby**. Final choice comes from a pilot: 3–5 people describe ~20 cars in their own words, and the 4 most common traits are kept.

**Participant split** (avoids people rating faces to stay consistent with the cars they already saw):

| | Group A (~half) | Group B (~half) |
|---|---|---|
| Part 1 | Rate **cars** on the 4 traits | Rate **generated faces** on the 4 traits (never see the cars) |
| Part 2 | Matching test | Matching test |

Trait ratings always come before the matching test, so nobody has seen car–face pairs while rating.

**Budget** (~15 min per person):
- ~40 cars for the main model; ~20 of those also go through the drawn-face model.
- Trait ratings: 40 items × 4 traits ≈ 8 min.
- Matching: ~40 trials (30 realistic + 10 drawn) ≈ 5 min.

**Analysis:**
- **Traits:** for each trait, average car ratings (Group A) and face ratings (Group B) per car, then correlate across the 40 cars (Spearman). With 40 cars, r ≈ 0.4 is significant. Check rater consistency by splitting each group in half and correlating the two halves.
- **Matching:** percent correct vs 50% chance (binomial test, or a mixed model with participant and car as random effects). With 15+ people × 30 trials, an effect of ~60% is detectable.
- **Realistic vs drawn:** compare matching accuracy and trait correlations on the shared ~20 cars. Treat as exploratory given the small sample.

**Participants:** personal project, so no formal ethics review, but surveys stay anonymous, with a short consent note and no personal data collected.

## Open questions

- **Survey access:** confirm participants can open and answer the artifact link (test with 1–2 people before the real study).
- **Grayscale vs color:** configs default to grayscale (avoids car color leaking into faces); revisit after the first run.
- **Loss / "from scratch":** does a pretrained network used *only inside the loss* (e.g. VGG perceptual loss) count as breaking "from scratch"? Default: no pretrained anything; start with L1.
