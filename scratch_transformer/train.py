"""Train a Transformer language model from scratch **with resume‑from‑checkpoint
and token‑dropout regularisation**.

Usage examples
--------------
# fresh run with external YAML hyper‑parameters
$ python -m scratch_transformer.train --config configs/base.yml

# resume training from an existing checkpoint
$ python -m scratch_transformer.train --config configs/base.yml \
        --init_ckpt checkpoints/epoch10.pt
"""
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
    # data
    data_dir: str  = "data/wikitext-2/cleaned"
    tok_file: str  = "data/tokenizer/v1/tokenizer.json"

    # model
    d_model: int    = 384
    num_layers: int = 6
    num_heads: int  = 6
    seq_len: int    = 512
    dropout: float  = 0.1
    token_dropout: float = 0.2     # probability to replace a token with <unk>

    # optimization

    batch: int      = 8
    grad_accum: int = 4
    lr: float      = 3e-4
    warmup: int     = 800
    weight_decay: float = 0.05

    epochs: int    = 30

    # checkpointing / resume
    ckpt_dir: str  = "checkpoints"
    init_ckpt: str | None = None    # path to checkpoint to resume from
    reset_scheduler: bool = False

# --------------------------------------------------------------------------- #
# 3 · Helper                                                                  #
# --------------------------------------------------------------------------- #


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def _move_optimizer_to_device(optim, device: str):
    """Move all tensors inside the optimizer state to the desired device."""
    for param_state in optim.state.values():
        for k, v in param_state.items():
            if isinstance(v, torch.Tensor):
                param_state[k] = v.to(device)


# --------------------------------------------------------------------------- #
# 4 · Main train loop                                                         #
# --------------------------------------------------------------------------- #


def train(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # ---------- tokenizer & datasets ---------- #
    tok = Tokenizer.from_file(cfg.tok_file)
    vocab = tok.get_vocab_size()
    unk_id = tok.token_to_id("<unk>")  # fallback for token dropout

    ds_train = TextBlockDataset(f"{cfg.data_dir}/train.txt", tok, cfg.seq_len)
    ds_val   = TextBlockDataset(f"{cfg.data_dir}/valid.txt", tok, cfg.seq_len)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch, shuffle=False, drop_last=False)

    # ---------- model ---------- #
    model = TransformerLM(
                vocab_size=vocab,
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                max_seq_len=cfg.seq_len,
            ).to(device)

    # ---------- optimizer & scheduler ---------- #
    # optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    # Optimizer with weight‑decay
    optim = torch.optim.AdamW(
                              model.parameters(),
                              lr=cfg.lr,
                              betas=(0.9, 0.95),
                              weight_decay=cfg.weight_decay,  # L2 regularization
                             )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optim,
    #     lambda step: min(1.0, step / cfg.warmup)
    #                  * 0.5 * (1 + math.cos(math.pi * step / (len(dl_train) * cfg.epochs)))
    # )

    total_steps = cfg.epochs * len(dl_train) // cfg.grad_accum

    # Cosine scheduler to zero over the whole run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                 optim,
                 T_max=total_steps,  # reach LR~0 at final step
                )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #              optim, T_0=total_steps//4, T_mult=2
    #             )

    scaler = torch.cuda.amp.GradScaler()
    # two separate losses
    loss_fn_train = nn.CrossEntropyLoss(label_smoothing=0.05)  # back-prop
    loss_fn_eval  = nn.CrossEntropyLoss()               # PPL metric

    ckpt_path = pathlib.Path(cfg.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    # ---------------------------------------------------------------------- #
    # Resume logic                                                           #
    # ---------------------------------------------------------------------- #
    start_epoch = 1
    if cfg.init_ckpt:
        ckpt_file = pathlib.Path(cfg.init_ckpt)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_file} not found")
        state = torch.load(ckpt_file, map_location="cpu")  # keep RNG on CPU
        model.load_state_dict(state["model"], strict=True)
        optim.load_state_dict(state["optim"])
        scheduler.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        _move_optimizer_to_device(optim, device)

        # optional LR override from config
        if cfg.lr is not None:
            for g in optim.param_groups:
                g["lr"] = cfg.lr
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [cfg.lr for _ in scheduler.base_lrs]

        # optionally rewind the scheduler’s internal clock
        if cfg.reset_scheduler:
            # Recompute horizon so cosine reaches eta_min at cfg.epochs
            steps_per_epoch = len(dl_train) // cfg.grad_accum
            new_T = steps_per_epoch * cfg.epochs   # number of scheduler steps we plan to run

            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.T_max = new_T
                scheduler.last_epoch = -1
                scheduler._step_count = 0

            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.T_0 = new_T
                scheduler.last_epoch = -1
                scheduler._step_count = 0
                scheduler.T_cur = 0

        # call once in any case so new LR sticks
        scheduler.step()  # apply fresh LR

        if "rng" in state:
            torch.random.set_rng_state(state["rng"])
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resumed from {ckpt_file} (epoch {start_epoch - 1})")

    global_step = 0
    train_iter = cycle(dl_train)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for _ in range(len(dl_train)):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            # Token dropout -----------------------------------------------
            if cfg.token_dropout > 0.0:
                # prevent special tokens (<eos>, <unk>) from being corrupted
                mask = (torch.rand_like(x.float()) < cfg.token_dropout) & (x != unk_id)
                x = x.masked_fill(mask, unk_id)

            with torch.cuda.amp.autocast():
                logits = model(x)                       # (B,L,V)
                loss   = loss_fn_train(logits.view(-1, vocab), y.view(-1)) / cfg.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (global_step + 1) % cfg.grad_accum == 0:
                # Gradient clipping - do this once per optimizer update
                scaler.unscale_(optim)                   # required before clipping when using amp
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
            global_step += 1

        # ---- metrics ---- #
        # train_ppl = math.exp(running_loss * cfg.grad_accum / len(dl_train))
        # val_ppl   = evaluate(model, dl_val, loss_fn, vocab, device)
        train_ppl = evaluate(model, dl_train, loss_fn_eval, vocab, device)
        val_ppl   = evaluate(model, dl_val, loss_fn_eval, vocab, device)
        elapsed   = time.time() - t0

        print(
            f"Epoch {epoch:02d} | train ppl {train_ppl:6.1f} | val ppl {val_ppl:6.1f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f} min"
        )

        # ---- persistence ---- #
        # torch.save(model.state_dict(), ckpt_path / f"epoch{epoch:02d}.pt")
        torch.save(
                   {
                    "epoch":   epoch,
                    "model":   model.state_dict(),
                    "optim":   optim.state_dict(),
                    "sched":   scheduler.state_dict(),
                    "scaler":  scaler.state_dict(),      # AMP GradScaler
                    "cfg":     cfg.__dict__,             # (optional) hyper-params
                    "rng":     torch.random.get_rng_state()  # (optional) full determinism
                    },
                    ckpt_path / f"epoch{epoch:02d}.pt"
                  )

# --------------------------------------------------------------------------- #
# 5 · Evaluate                                                              #
# --------------------------------------------------------------------------- #


def evaluate(model, dl, loss_fn, vocab, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            # logits = model(x)
            # loss   = loss_fn(logits.view(-1, vocab), y.view(-1))
            # total += loss.item() * x.size(0)
            total += loss_fn(model(x).view(-1,vocab), y.view(-1)).item()*x.size(0)
            n += x.size(0)
    return math.exp(total / n)


# --------------------------------------------------------------------------- #
# 6 · Config loader (JSON **or** YAML)                                        #
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
