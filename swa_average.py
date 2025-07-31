import torch, pathlib, sys
from scratch_transformer.model import TransformerLM                        # same model code
from scratch_transformer.train import Config, evaluate, TextBlockDataset   # reuse your eval fn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

# ---- config must match training run ----
cfg = Config()
cfg.d_model    = 512      # or 384 if you later downsized
cfg.num_heads  = 8
cfg.num_layers = 6
cfg.seq_len    = 512
cfg.tok_file   = "data/tokenizer/v1/tokenizer.json"

# ---- load tokenizer & eval set for final perplexity ----
from tokenizers import Tokenizer
tok = Tokenizer.from_file(cfg.tok_file)
# ds_val = ...  # reuse your TextBlockDataset call
# dl_val = ...  # DataLoader(batch=6, etc.)
ds_val   = TextBlockDataset(f"{cfg.data_dir}/valid.txt", tok, cfg.seq_len)
dl_val   = DataLoader(ds_val,   batch_size=cfg.batch, shuffle=False, drop_last=False)

# ---- list your ckpts here ----
ckpts = [
    "checkpoints_ft_3/epoch08.pt",
    "checkpoints_ft_3/epoch09.pt",
    "checkpoints_ft_3/epoch11.pt",
    "checkpoints_ft_3/epoch15.pt",
    "checkpoints_ft_4/epoch09.pt",
]

# ---- build model skeleton and AveragedModel ----
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = TransformerLM(
    vocab_size = tok.get_vocab_size(),
    d_model    = cfg.d_model,
    num_layers = cfg.num_layers,
    num_heads  = cfg.num_heads,
    dropout    = 0.0,            # irrelevant for eval
    max_seq_len= cfg.seq_len,
).to(device)

swa_model = AveragedModel(base_model)

# ---- merge checkpoints ----
for path in ckpts:
    state = torch.load(path, map_location="cpu")

    temp_model = TransformerLM(
        vocab_size = tok.get_vocab_size(),
        d_model    = cfg.d_model,
        num_layers = cfg.num_layers,
        num_heads  = cfg.num_heads,
        dropout    = 0.0,
        max_seq_len= cfg.seq_len,
    ).to(device)
    temp_model.load_state_dict(state["model"])
    swa_model.update_parameters(temp_model)     # pass the model

    print(f"merged {path}")

# ---- save averaged weights ----
out = pathlib.Path("checkpoints_swa/epoch_swa.pt")
out.parent.mkdir(exist_ok=True)
torch.save({"model": swa_model.module.state_dict()}, out)
print("Saved averaged model to", out)

# ---- evaluate perplexity ----
loss_fn = torch.nn.CrossEntropyLoss()
ppl = evaluate(swa_model.module, dl_val, loss_fn,
               tok.get_vocab_size(), device)
print(f"SWA validation PPL: {ppl:.2f}")
