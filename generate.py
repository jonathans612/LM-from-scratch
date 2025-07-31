# generate.py
import torch, sys, readline  # readline → arrow-key history
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from scratch_transformer.model import TransformerLM

ckpt_path = "final_model/wt2_512d6L_SWA167.pt"
tok_path  = "final_model/tokenizer.json"
seq_len   = 512
device    = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- load ----------
tok = Tokenizer.from_file(tok_path)
tok.decoder = ByteLevel()  # decoder fixes some tokenization artifacts
vocab = tok.get_vocab_size()
state = torch.load(ckpt_path, map_location="cpu")

model = TransformerLM(
    vocab_size=vocab, d_model=512, num_layers=6,
    num_heads=8, dropout=0.0, max_seq_len=seq_len
).to(device).eval()
model.load_state_dict(state["model"])

def apply_repetition_penalty(logits, ids, penalty=1.2):
    if penalty == 1.0:
        return logits
    for t in set(ids):
        logits[0, t] /= penalty   # divide log-prob - lowers softmax prob
    return logits

def sample(prompt, max_new=40, temp=0.6, top_k=50, top_p=0.9, rep_pen=1.2):
    ids = tok.encode(prompt).ids
    model.eval()
    for _ in range(max_new):
        x = torch.tensor(ids[-seq_len:], device=device)[None]
        with torch.no_grad():
            logits = model(x)[:, -1] / temp
            logits = apply_repetition_penalty(logits, ids, rep_pen)
            probs = F.softmax(logits, dim=-1)

            # top-k filter
            if top_k:
                top_vals, top_idx = probs.topk(top_k)
                probs = torch.zeros_like(probs).scatter_(-1, top_idx, top_vals)
                probs /= probs.sum()

            # nucleus (top-p) filter
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[..., 0] = True                 # always keep most probable
            filtered_probs = torch.zeros_like(probs).scatter_(
                -1, sorted_idx, sorted_probs * keep_mask
            )
            filtered_probs /= filtered_probs.sum()   # renormalise

            next_id = torch.multinomial(filtered_probs, num_samples=1).item()
        ids.append(next_id)
        if next_id == tok.token_to_id("<eos>"):
            break
    return tok.decode(ids, skip_special_tokens=True)

# ---------- REPL ----------
while True:
    prompt = input("\n» ").strip()
    if not prompt:
        continue
    print(sample(prompt))
