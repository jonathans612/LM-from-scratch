Language-Model-from-Scratch · RTX 2070 Edition
==============================================

## 1 · Executive Summary
Build and publish a small GPT-style language model as both a résumé showcase and a personal research exercise.  
Target: train a decoder-only Transformer on Wikipedia text and achieve ≤ 40 validation perplexity on the WikiText-2 benchmark using a single NVIDIA RTX 2070 (8 GB).

---

## 2 · Problem Definition
| Field                   | Decision |
|-------------------------|----------|
| **Task type**           | Autoregressive next-token prediction |
| **Domain**              | English Wikipedia prose |
| **Sequence length**     | 512 tokens |
| **Out-of-scope**        | No RLHF, no multilingual, no code modeling |

---

## 3 · Success Metrics
* **Primary:** Validation perplexity ≤ 40 on WikiText-2.  
* **Secondary:** Manual quality check of generated samples for coherence and absence of glaring toxicity.

---

## 4 · Dataset & Licensing
| Item                            | Choice / Notes |
|---------------------------------|----------------|
| **Primary dataset**             | WikiText-2 (`~2 M` tokens, CC-BY-SA 3.0) |
| **Optional extension**          | 100 MB slice of OpenWebText (public-domain or CC sources only) *after* pipeline is stable |
| **Filter rules**                | Strip HTML / non-printable Unicode, drop docs outside 10 – 2048 tokens, basic profanity filter |
| **Download script**             | `scripts/get_data.sh` |
| **License compatibility**       | All data licenses compatible with MIT code license |

---

## 5 · Pre-processing & Tokenization
| Item            | Decision |
|-----------------|----------|
| **Tokenizer**   | Byte-Pair Encoding (BPE) via HF `tokenizers` |
| **Vocab size**  | 16 000 tokens (bump to 32 k if adding OpenWebText) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>` |
| **Versioning**  | Save `data/tokenizer/v1/tokenizer.json`; record SHA-256 in README |

---

## 6 · Model Specification (fits 8 GB)
| Hyper-parameter | Value | Rationale |
|-----------------|-------|-----------|
| Layers (`n_layers`) | **6** | Keeps depth benefits |
| Width (`d_model`)   | **384** | Balances capacity vs. VRAM |
| Heads (`n_heads`)   | 6 (64-dim each) | `d_model / 64` rule |
| Feed-forward (`d_ff`)| 1536 (≈4×d_model) | Transformer convention |
| **Dropout**         | 0.1 | Standard for small LMs |
| Weight init         | Xavier-Uniform | Stable |

Total parameters ≈ **46 M**.

---

## 7 · Training Protocol
| Item                      | Setting |
|---------------------------|---------|
| **Optimizer**             | AdamW (β = 0.9/0.95, ε = 1e-8, wd = 0.01) |
| **Learning rate**         | 3 × 10⁻⁴ base |
| **Schedule**              | Linear warm-up 800 steps → cosine decay to 0.1 · LR |
| **Batching**              | Physical: 8 × 512 tokens (≈4 k) <br>Grad-accum × 4 → *effective* 16 k tokens |
| **Precision**             | AMP (fp16) |
| **Gradient clip**         | 1.0 (global L2 norm) |
| **Activation checkpointing** | Enabled (PyTorch `torch.utils.checkpoint`) |
| **Checkpoint cadence**    | Every 1 000 steps; retain last 3 |

---

## 8 · Hardware & Runtime Budget
| Resource             | Plan |
|----------------------|------|
| **GPU**              | Local RTX 2070 (8 GB) |
| **Expected wall-time** | ~12 h per full 30-epoch run on WikiText-2; entire weekend (~36 h clock) for extended corpus |
| **Contingency**      | Reduce sequence to 256 or use Colab Pro if VRAM issues persist |

---

## 9 · Experiment Tracking & Reproducibility
| Aspect          | Decision |
|-----------------|----------|
| **Logger**      | Weights & Biases (free tier) |
| **Seeds**       | `torch`, `numpy`, `random` all set to 42 |
| **Determinism** | `torch.backends.cudnn.deterministic=True`, `benchmark=False` |
| **Config snapshot** | YAML + git commit hash auto-saved per run (`wandb.save()`) |

---

## 10 · Deliverables
| Artifact                      | Location / Format |
|-------------------------------|-------------------|
| Training script               | `scratch_transformer/train.py` |
| Best checkpoint               | `releases/wikitext2_46M.pt` |
| Tokenizer file                | `data/tokenizer/v1/tokenizer.json` |
| Sampling CLI                  | `python -m scratch_transformer.cli generate …` |
| Colab demo notebook (optional)| `notebooks/demo.ipynb` |
| README with quick-start guide | `README.md` |

*Code will be public from day 1; weights released only when success metric is hit.*

---

## 11 · Timeline & Milestones
| Week | Goal | Definition of Done |
|------|------|--------------------|
| 1 | **Data + tokenizer** | `get_data.sh` + `train_tokenizer.py` produce SHA-logged outputs |
| 2 | **Core layers & tests** | `pytest` green; CI passes |
| 3 | **First end-to-end epoch** | Loss decreases; val ppl logged |
| 4 | **Full training run** | ≤ 40 ppl on WikiText-2 |
| 5 | **Docs & release** | README polished; v0.1.0 tag + checkpoint uploaded |

---

## 12 · Risks & Mitigation
| Risk                     | Mitigation |
|--------------------------|------------|
| GPU OOM                  | Seq = 256, grad checkpointing, reduce batch |
| Over-fitting (tiny data) | Early-stop, dropout ↑, add OpenWebText slice |
| Hardware outage          | Auto-resume from last checkpoint |
| Motivation drift         | Weekly status notes in README |

---

## 13 · Maintenance & Support
| Item          | Policy |
|---------------|--------|
| **License**   | MIT |
| **Versioning**| Semantic Versioning (`v0.1.0` first public release) |
| **Issue policy** | Accept bug reports; close usage questions with doc links |
| **Community** | Enable GitHub Discussions when project stabilizes |

---

## 14 · Communication / Documentation
*Phase 1:* Basic README (install, train, sample, results).  
*Phase 2:* After hitting the metric, add a Colab demo and auto-generated API docs via MkDocs (optional).

---
