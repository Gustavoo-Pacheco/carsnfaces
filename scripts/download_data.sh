#!/usr/bin/env bash
# Download the datasets (no approval or request forms needed).
#
#   bash scripts/download_data.sh [celeba|cartoon|cars|all] [DATA_DIR]
#
# CelebA and Stanford Cars come from Kaggle: needs a free Kaggle account and an API
# token at ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
# On Colab, notebooks/01_train_autoencoder.ipynb sets this up.
set -euo pipefail

WHAT="${1:-all}"
DATA="${2:-${CNF_DATA:-data}}"
RAW="$DATA/raw"
mkdir -p "$RAW"

celeba() {
  echo "== CelebA (aligned & cropped, ~200k images, ~1.4 GB)"
  kaggle datasets download -d jessicali9530/celeba-dataset -p "$RAW/celeba_zip"
  unzip -q -o "$RAW/celeba_zip/"*.zip -d "$RAW/celeba_zip"
  # keep only the image folder
  mkdir -p "$RAW/celeba"
  find "$RAW/celeba_zip" -name '*.jpg' -exec mv -t "$RAW/celeba" {} + 2>/dev/null \
    || find "$RAW/celeba_zip" -name '*.jpg' -exec mv {} "$RAW/celeba/" \;
  rm -rf "$RAW/celeba_zip"
  echo "   $(ls "$RAW/celeba" | wc -l) images in $RAW/celeba"
}

cartoon() {
  echo "== Google Cartoon Set (10k, CC-BY 4.0)"
  # Source: https://google.github.io/cartoonset/
  # The bucket needs a signed-in Google account (no approval). On Colab, run
  # `from google.colab import auth; auth.authenticate_user()` first.
  # Locally: `gcloud auth login`, or download the file in a signed-in browser and put
  # it at $RAW/cartoonset10k.tgz before running this.
  mkdir -p "$RAW/cartoonset"
  if [ ! -f "$RAW/cartoonset10k.tgz" ]; then
    gsutil cp gs://cartoonset_public_files/cartoonset10k.tgz "$RAW/cartoonset10k.tgz"
  fi
  tar -xzf "$RAW/cartoonset10k.tgz" -C "$RAW/cartoonset" --strip-components=1
  rm "$RAW/cartoonset10k.tgz"
  find "$RAW/cartoonset" -name '*.csv' -delete  # attribute files, not needed
  echo "   $(ls "$RAW/cartoonset" | wc -l) images in $RAW/cartoonset"
}

cars() {
  echo "== Stanford Cars (~16k images)"
  kaggle datasets download -d eduardo4jesus/stanford-cars-dataset -p "$RAW/stanford_cars_zip"
  unzip -q -o "$RAW/stanford_cars_zip/"*.zip -d "$RAW/stanford_cars"
  rm -rf "$RAW/stanford_cars_zip"
  echo "   extracted to $RAW/stanford_cars"
}

case "$WHAT" in
  celeba) celeba ;;
  cartoon) cartoon ;;
  cars) cars ;;
  all) celeba; cartoon; cars ;;
  *) echo "usage: $0 [celeba|cartoon|cars|all] [DATA_DIR]"; exit 1 ;;
esac
