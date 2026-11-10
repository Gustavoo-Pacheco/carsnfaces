# Cars n' Faces

**Do neural networks see faces in cars the way people do?**

People see faces in car fronts: headlights as eyes, the grille as a mouth, and some cars look *angry*, *sad* or *chubby*. This is **pareidolia**. This project trains an autoencoder **only on human faces**, then feeds it **pictures of cars**. Because the model can only produce faces, it is forced to "find" a face in each car. A human study then tests whether the model finds the **same face people see**.

```
Training:  face ──► encoder ──► latent ──► decoder ──► face
Testing:   car  ──► encoder ──► latent ──► decoder ──► face (the face "seen" in the car)
```

> **Status:** scaffolding only. The code is a first draft and has **not been run or tested yet**; no data has been downloaded and no models trained. See [`PLAN.md`](PLAN.md) for the next steps.

## Research questions

1. **Trait alignment (main):** if people see a car as angry, sad, happy or chubby, does the model's face carry the same trait?
2. **Matching:** shown a car and two generated faces, can people tell which face came from that car?
3. **Realistic vs drawn (secondary):** does a model trained on cartoon faces match human pareidolia better or worse than one trained on real faces?

## Approach

| | |
|---|---|
| **Main model** | Convolutional autoencoder, built from scratch, trained on real faces (CelebA) |
| **Secondary model** | Same architecture, trained on drawn faces (Google Cartoon Set) |
| **Input cars** | ~40 hand-picked car fronts from Stanford Cars (plus rears as controls) |
| **Human study** | 15–30 participants, web survey: 4-trait ratings + matching test |
| **Resolution** | 64px first, then 128px |

Key design choices:
- **Augmentation** (random crops, shifts, rotations): face datasets are pre-aligned, so without it the model would draw eyes at fixed pixels regardless of the car.
- **Grayscale by default:** keeps car color from leaking into the faces.
- **Split participant groups:** one group rates cars, the other rates the generated faces without seeing the cars, so ratings can't be copied between them.

Full rationale, design and open questions: [`CONTEXT.md`](CONTEXT.md).

## Stack

- **Google Colab Pro**: GPU training
- **PyTorch**: models written from scratch, no pretrained networks
- **Weights & Biases** (free plan): loss curves, face reconstructions and car→face samples
- **Google Drive**: datasets, checkpoints, outputs
- **Claude artifact**: the survey website and its response storage

## Repository layout

```
├── CONTEXT.md                  idea, decisions, open questions
├── PLAN.md                     step-by-step plan with checkboxes
├── configs/                    one YAML per experiment
├── src/carsnfaces/
│   ├── config.py               YAML → Config
│   ├── data.py                 image dataset + augmentation
│   ├── model.py                convolutional autoencoder
│   ├── train.py                training loop, W&B logging, auto-resume
│   └── infer.py                cars → faces
├── scripts/
│   ├── download_data.sh        CelebA / Cartoon Set / Stanford Cars
│   └── prepare_images.py       select, square and resize images
├── notebooks/                  Colab notebooks (train, cars → faces)
├── data/selections/            hand-picked car lists
├── survey/                     human study (artifact)
└── analysis/                   statistics on survey results
```

## Quickstart (Colab)

1. Copy the repo to Google Drive at `MyDrive/carsnfaces`.
2. Put your Kaggle API token at `MyDrive/carsnfaces/secrets/kaggle.json`.
3. Open [`notebooks/01_train_autoencoder.ipynb`](notebooks/01_train_autoencoder.ipynb) with a GPU runtime and run the cells: setup → credentials → data download → smoke test → training.
4. Open [`notebooks/02_cars_to_faces.ipynb`](notebooks/02_cars_to_faces.ipynb) to turn car fronts into faces.

Command line equivalent:

```bash
pip install -e . -r requirements.txt
bash scripts/download_data.sh celeba
python -m carsnfaces.train --config configs/faces_64.yaml
python -m carsnfaces.infer --ckpt runs/faces_64/final.pt \
    --input data/processed/car_fronts_64 --out outputs/faces_64/car_fronts
```

Training resumes automatically from `runs/<name>/last.pt`. Override any config value with `--set key=value`.

## Datasets

All usable without an approval process:

| Dataset | Use | Source |
|---|---|---|
| CelebA | Real faces (main model) | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) |
| Google Cartoon Set | Drawn faces (secondary model) | [google.github.io/cartoonset](https://google.github.io/cartoonset/) (CC-BY 4.0; needs Google sign-in) |
| Stanford Cars | Car fronts and rears | [Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) |

Datasets are not redistributed in this repo; each keeps its own license and terms.

## Related work

- Hamilton et al. (2024). *Seeing Faces in Things: A Model and Dataset for Pareidolia.* ECCV.
- Windhager et al. (2008). *Face to face: The perception of automotive designs.* Human Nature.
- Kühn et al. (2014). *Is this car looking at you? How anthropomorphism predicts fusiform face area activation when seeing cars.* PLoS ONE.
- Wardle et al. (2020). *Rapid and dynamic processing of face pareidolia in the human brain.* Nature Communications.
- Dobs et al. (2022). *Brain-like functional specialization emerges spontaneously in deep neural networks.* Science Advances.
