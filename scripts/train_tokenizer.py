#!/usr/bin/env python
"""
Train a byte-pair-encoding (BPE) tokenizer on the cleaned WikiText-2 corpus.

Outputs:
  data/tokenizer/v1/tokenizer.json   ← model + merges
  data/tokenizer/v1/meta.yaml        ← reproducibility metadata
"""

import argparse, hashlib, pathlib, yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# --------------------------------------------------------------------------- #
# 1 · CLI                                                                     #
# --------------------------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/wikitext-2/cleaned")
p.add_argument("--out_dir",  default="data/tokenizer/v1")
p.add_argument("--vocab_size", type=int, default=16_000)
args = p.parse_args()

data_dir = pathlib.Path(args.data_dir)
out_dir  = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2 · Gather training text & compute SHA                                      #
# --------------------------------------------------------------------------- #
train_file = data_dir / "train.txt"
sha = hashlib.sha256(train_file.read_bytes()).hexdigest()

# --------------------------------------------------------------------------- #
# 3 · Build & train tokenizer                                                 #
# --------------------------------------------------------------------------- #
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
    min_frequency=2,
)
tokenizer.train([str(train_file)], trainer=trainer)

# --------------------------------------------------------------------------- #
# 4 · Save                                                                    #
# --------------------------------------------------------------------------- #
tok_path = out_dir / "tokenizer.json"
tokenizer.save(str(tok_path))

(out_dir / "meta.yaml").write_text(
    yaml.safe_dump(
        dict(
            vocab_size=args.vocab_size,
            corpus_sha256=sha,
            special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
        )
    )
)

print(f"Tokenizer saved to {tok_path}")
