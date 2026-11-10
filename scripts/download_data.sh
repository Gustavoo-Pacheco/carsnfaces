#!/usr/bin/env bash
# Download the datasets (no approval or request forms needed).
#
#   bash scripts/download_data.sh [celeba|dvm|dvm-full|cartoon] [DEST_DIR]
#
# Downloads go to DEST_DIR (default: /content/raw on Colab's local disk — fast, and
# the raw files are not needed after preprocessing; only the processed arrays go to Drive).
#   celeba    CelebA aligned & cropped via kagglehub (public dataset, no Kaggle token needed)   ~1.4 GB
#   dvm       DVM-CAR quality-checked fronts + tables (Figshare, CC BY-NC 4.0)                 ~0.9 GB
#   dvm-full  DVM-CAR full resized image set, needed only to pick rear views                    ~14.6 GB
#   cartoon   Google Cartoon Set 10k (needs a signed-in Google account; phase 5 only)
set -euo pipefail

WHAT="${1:?usage: $0 [celeba|dvm|dvm-full|cartoon] [DEST_DIR]}"
DEST="${2:-/content/raw}"
mkdir -p "$DEST"

fetch() {  # url file
  if [ -s "$DEST/$2" ]; then echo "   $2 already there"; return; fi
  curl -L --fail --retry 5 -C - -o "$DEST/$2.part" "$1" && mv "$DEST/$2.part" "$DEST/$2"
}

case "$WHAT" in
  celeba)
    echo "== CelebA (Kaggle jessicali9530/celeba-dataset)"
    python -c "import kagglehub; print(kagglehub.dataset_download('jessicali9530/celeba-dataset'))" \
      | tail -1 > "$DEST/celeba_path.txt"
    echo "   path: $(cat "$DEST/celeba_path.txt")"
    ;;
  dvm)
    echo "== DVM-CAR fronts + tables"
    fetch https://ndownloader.figshare.com/files/34792480 Confirmed_fronts.zip
    fetch https://ndownloader.figshare.com/files/38754867 tables_V2.0.zip
    ;;
  dvm-full)
    echo "== DVM-CAR full image set (14.6 GB, for rear views)"
    fetch https://ndownloader.figshare.com/files/34792453 resized_DVM_v2.zip
    ;;
  cartoon)
    echo "== Google Cartoon Set 10k (run google.colab.auth.authenticate_user() first)"
    gsutil cp gs://cartoonset_public_files/cartoonset10k.tgz "$DEST/cartoonset10k.tgz"
    ;;
  *) echo "unknown: $WHAT"; exit 1 ;;
esac
