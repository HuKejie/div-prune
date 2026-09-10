"""Shape tests for the Transformer MT model and attention-matrix collection."""

import torch
from omegaconf import OmegaConf

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "name": "transformer_mt",
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "src_vocab": 100,
            "tgt_vocab": 100,
            "pad_idx": 3,
        }
    )


def _tensors():
    src = torch.randint(4, 100, (2, 6))
    tgt = torch.randint(4, 100, (2, 5))
    src_pad = src == 3
    tgt_pad = tgt == 3
    causal = torch.zeros(5, 5, dtype=torch.bool)
    return src, tgt, src_pad, tgt_pad, causal


def test_forward_shape_and_attn_collection():
    model = ModelFactory("transformer_mt")(_cfg())
    src, tgt, src_pad, tgt_pad, causal = _tensors()

    out = model(src, tgt, src_pad, tgt_pad, causal)
    assert out["logits"].shape == (2, 5, 100)

    out = model(src, tgt, src_pad, tgt_pad, causal, collect_attn=True)
    # per layer: 1 enc self + 1 dec self + 1 cross
    assert len(out["attn_mats"]) == 2 * 3
    for w in out["attn_mats"]:
        assert w.shape[0] == 2 and w.shape[1] == 4


def test_encode_decode_consistent_with_forward():
    model = ModelFactory("transformer_mt")(_cfg())
    src, tgt, src_pad, tgt_pad, causal = _tensors()
    with torch.no_grad():
        logits1 = model(src, tgt, src_pad, tgt_pad, causal)["logits"]
        memory = model.encode(src, src_pad)
        logits2 = model.decode(memory, tgt, tgt_pad, src_pad, causal)
    assert torch.allclose(logits1, logits2, atol=1e-6)
