# scratch_transformer/train.py
from __future__ import annotations
import argparse, math, pathlib, time, json
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model  import TransformerLM
from .layers import causal_mask
from tokenizers import Tokenizer


# --------------------------------------------------------------------------- #
# 1 · Data set + loader                                                       #
# --------------------------------------------------------------------------- #
class TextBlockDataset(Dataset):
    """
    Streams fixed-length token blocks from a plain-text file.

    Each __getitem__ returns (input_ids, target_ids) where
    target == input shifted left by one w/ <eos> at the end.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        tokenizer: Tokenizer,
        seq_len: int = 512,
    ):
        self.seq_len   = seq_len
        self.tokenizer = tokenizer
        self.toks: list[int] = tokenizer.encode(
            pathlib.Path(path).read_text(encoding="utf8")
        ).ids

        # chop into non-overlapping seq_len blocks
        n_blocks = len(self.toks) // seq_len
        self.toks = self.toks[: n_blocks * seq_len]

    def __len__(self) -> int:
        return len(self.toks) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end   = start + self.seq_len
        x = torch.tensor(self.toks[start:end], dtype=torch.long)
        y = torch.roll(x, shifts=-1)
        y[-1] = self.tokenizer.token_to_id("<eos>")
        return x, y


# --------------------------------------------------------------------------- #
# 2 · Training utilities                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_dir: str  = "data/wikitext-2/cleaned"
    tok_file: str  = "data/tokenizer/v1/tokenizer.json"
    seq_len: int   = 512
    batch: int     = 8
    grad_accum: int = 4
    d_model: int   = 384
    num_heads: int = 6
    num_layers: int = 6
    dropout: float = 0.1
    lr: float      = 3e-4
    warmup: int    = 800
    epochs: int    = 30
    ckpt_dir: str  = "checkpoints"

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# --------------------------------------------------------------------------- #
# 3 · Main train loop                                                         #
# --------------------------------------------------------------------------- #
def train(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(cfg.tok_file)
    vocab = tok.get_vocab_size()

    ds_train = TextBlockDataset(f"{cfg.data_dir}/train.txt", tok, cfg.seq_len)
    ds_val   = TextBlockDataset(f"{cfg.data_dir}/valid.txt", tok, cfg.seq_len)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch, shuffle=False, drop_last=False)

    model = TransformerLM(
        vocab_size=vocab,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_seq_len=cfg.seq_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda step: min(1.0, step / cfg.warmup)
                     * 0.5 * (1 + math.cos(math.pi * step / (len(dl_train) * cfg.epochs)))
    )
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()

    ckpt_path = pathlib.Path(cfg.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    global_step = 0
    train_iter = cycle(dl_train)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for _ in range(len(dl_train)):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x)                       # (B,L,V)
                loss   = loss_fn(logits.view(-1, vocab), y.view(-1)) / cfg.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (global_step + 1) % cfg.grad_accum == 0:
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
            global_step += 1

        train_ppl = math.exp(running_loss * cfg.grad_accum / len(dl_train))
        val_ppl   = evaluate(model, dl_val, loss_fn, vocab, device)
        elapsed   = time.time() - t0

        print(
            f"Epoch {epoch:02d} | train ppl {train_ppl:6.1f} | val ppl {val_ppl:6.1f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f} min"
        )

        torch.save(model.state_dict(), ckpt_path / f"epoch{epoch:02d}.pt")

def evaluate(model, dl, loss_fn, vocab, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits.view(-1, vocab), y.view(-1))
            total += loss.item() * x.size(0)
            n += x.size(0)
    return math.exp(total / n)

# --------------------------------------------------------------------------- #
# 4 · Config loader (JSON **or** YAML)                                        #
# --------------------------------------------------------------------------- #

import json, yaml, pathlib

def load_cfg(path: pathlib.Path) -> dict:
    """Load a config file that may be .json, .yml, or .yaml."""
    text = path.read_text()
    if path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Transformer LM from scratch")
    p.add_argument("--config", type=pathlib.Path, help="Optional JSON config file", default=None)
    args = p.parse_args()

    cfg = Config()
    if args.config and args.config.exists():
        cfg.__dict__.update(load_cfg(args.config))

    train(cfg)
