# tests/test_model.py
import torch
from scratch_transformer.model import TransformerLM


def test_forward_backward():
    m = TransformerLM(vocab_size=100, d_model=384, num_layers=2, num_heads=6)
    inp = torch.randint(0, 100, (2, 8))
    out = m(inp)
    loss = out.mean()
    loss.backward()
    assert out.shape == (2, 8, 100)
