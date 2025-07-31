# eval_ppl.py
print("eval_ppl.py launched...", flush=True)
import argparse, torch, pathlib
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

# --- repo imports ---------------------------------------------------------
from scratch_transformer.model import TransformerLM
from scratch_transformer.train import Config, TextBlockDataset, evaluate
# --------------------------------------------------------------------------

def main(ckpt_path: pathlib.Path):
    # ---- load training hyper-params (adjust if different) -----------------
    cfg = Config()
    cfg.seq_len   = 512
    cfg.batch     = 6                 # eval batch; raise if you have VRAM
    cfg.d_model   = 512               # MUST match checkpoint
    cfg.num_heads = 8
    cfg.num_layers= 6
    # ----------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer
    tok = Tokenizer.from_file(cfg.tok_file)
    vocab = tok.get_vocab_size()

    # build model skeleton & load weights
    model = TransformerLM(
        vocab_size = vocab,
        d_model    = cfg.d_model,
        num_layers = cfg.num_layers,
        num_heads  = cfg.num_heads,
        dropout    = 0.0,
        max_seq_len= cfg.seq_len,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # dataset / loader for the *test* split
    ds_test = TextBlockDataset(f"{cfg.data_dir}/test.txt", tok, cfg.seq_len)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch*2, shuffle=False)  # 2× batch for speed

    # cross-entropy → perplexity
    loss_fn = torch.nn.CrossEntropyLoss()
    ppl = evaluate(model, dl_test, loss_fn, vocab, device)

    print(f"Test perplexity for {ckpt_path.name}: {ppl:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate PPL on WikiText-2 test split")
    p.add_argument("ckpt", type=pathlib.Path, help="Path to .pt checkpoint (model or swa)")
    args = p.parse_args()
    main(args.ckpt)
