#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download and clean WikiText‑2 (raw variant) via Hugging Face Datasets.
# Produces:
#   data/wikitext-2/cleaned/{train,valid,test}.txt
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="data/wikitext-2/cleaned"
mkdir -p "$ROOT"

python - <<'PY'
"""Fetch WikiText-2 (raw) with Hugging Face Datasets, clean minimal whitespace, save TSVs."""
import re, unicodedata, pathlib
from datasets import load_dataset

def _tidy(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text if text else None

root = pathlib.Path("data/wikitext-2/cleaned")
root.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

for split in ("train", "validation", "test"):
    out_file = root / ("valid.txt" if split == "validation" else f"{split}.txt")
    with out_file.open("w", encoding="utf8") as f:
        for line in ds[split]["text"]:
            clean = _tidy(line)
            if clean:
                f.write(clean + "\n")
    print(f"saved {out_file.relative_to(root.parent)} → {out_file.stat().st_size/1e6:.2f} MB")
PY

echo "WikiText-2 cleaned files are in $ROOT"