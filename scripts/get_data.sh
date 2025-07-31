#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download and clean WikiText‑2 (raw variant) via Hugging Face Datasets.
# Produces:
#   data/wikitext-2/cleaned/{train,valid,test}.txt
# -----------------------------------------------------------------------------
set -euo pipefail  # Exit on error, unset variable, or failed pipe

ROOT="data/wikitext-2/cleaned"  # Output directory for cleaned data
mkdir -p "$ROOT"                # Create output directory if it doesn't exist

python - <<'PY'
"""Fetch WikiText-2 (raw) with Hugging Face Datasets, clean minimal whitespace, save TSVs."""
import re, unicodedata, pathlib
from datasets import load_dataset

def _tidy(text: str) -> str | None:
    # Normalize unicode and collapse whitespace
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text if text else None  # Return None if line is empty

root = pathlib.Path("data/wikitext-2/cleaned")
root.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")  # Download dataset

for split in ("train", "validation", "test"):
    # Map 'validation' split to 'valid.txt', others to '{split}.txt'
    out_file = root / ("valid.txt" if split == "validation" else f"{split}.txt")
    with out_file.open("w", encoding="utf8") as f:
        for line in ds[split]["text"]:
            clean = _tidy(line)  # Clean each line
            if clean:
                f.write(clean + "\n")  # Write cleaned line if not empty
    # Print relative path and file size in MB
    print(f"saved {out_file.relative_to(root.parent)} → {out_file.stat().st_size/1e6:.2f} MB")
PY

echo "WikiText-2 cleaned files are in $ROOT"  # Notify user of output location
