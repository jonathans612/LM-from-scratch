import torch
from scratch_transformer.layers import DecoderBlock, causal_mask


def test_shape_roundtrip():
    blk = DecoderBlock(d_model=384, num_heads=6, dropout=0.0)
    x  = torch.randn(2, 8, 384)
    assert blk(x).shape == x.shape


def test_causal_mask():
    # grab the (4×4) mask from the (1×1×4×4) output
    m = causal_mask(4)[0, 0]                 # shape (4,4)
    # only the strict upper-triangle entries should be -inf
    upper_idx = torch.triu(torch.ones_like(m), diagonal=1).bool()
    assert torch.isneginf(m[upper_idx]).all()

    # and everything else should be 0
    for i in range(4):
        for j in range(4):
            if j > i:
                assert m[i, j] == float("-inf")
            else:
                assert m[i, j] == 0.0