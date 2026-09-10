"""Training loop for the Transformer/Multi30k leg, with the adaptive controller."""

import csv
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sacrebleu import corpus_bleu

from divprune.data_module.dataset.multi30k import make_batch
from divprune.trainer_module.adaptive_lambda import DivergenceController
from divprune.trainer_module.regularizer import transformer_attention_divergence

logger = logging.getLogger(__name__)


def make_pad_mask(ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """(B, T) boolean key-padding mask (True = ignore)."""
    return ids == pad_idx


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """2D (T, T) boolean causal mask (True = ignore)."""
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_pad: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_idx: int,
    max_len: int = 80,
) -> torch.Tensor:
    """Greedy decode for a batch; returns token ids including BOS."""
    model.eval()
    memory = model.encode(src, src_pad)
    ys = torch.full((src.size(0), 1), bos_id, device=src.device, dtype=torch.long)
    for _ in range(max_len):
        tgt_pad = make_pad_mask(ys, pad_idx)
        causal = make_causal_mask(ys.size(1), src.device)
        logits = model.decode(memory, ys, tgt_pad, src_pad, causal)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if bool((next_tok == eos_id).all()):
            break
    return ys


def _bleu(model, pairs, data, device, subset, sp) -> float:
    """Greedy-decoding BLEU on the first `subset` pairs."""
    pad, bos, eos = data["pad_idx"], data["bos_id"], data["eos_id"]
    hyps: List[str] = []
    refs: List[List[str]] = []
    pairs = pairs[:subset]
    for i in range(0, len(pairs), 64):
        src, _ = make_batch(pairs[i : i + 64], pad, device)
        src_pad = make_pad_mask(src, pad)
        ys = greedy_decode(model, src, src_pad, bos, eos, pad)
        for j, row in enumerate(ys):
            ids = [int(t) for t in row.tolist() if int(t) not in (pad, bos, eos)]
            hyps.append(sp.DecodeIds(ids))
            refs.append([pairs[i + j][1][1:-1] and sp.DecodeIds(pairs[i + j][1][1:-1]) or ""])
    return float(corpus_bleu(hyps, refs, tokenize="13a").score)


def train_transformer(
    cfg: DictConfig,
    model: nn.Module,
    data: Dict,
    trace_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Train the transformer with optional adaptive divergence regularization.

    Controller input is measured deterministically (eval mode) on the first
    batch of each epoch; the regularizer gradient is applied to that same
    batch inside the training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_pairs = data["train"]
    pad = data["pad_idx"]
    sp = data["sp"]
    batch_size = int(cfg.model.batch_size)
    val_subset = int(cfg.model.val_subset)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    use_reg = cfg.experiment.regularizer != "none"
    lam_fixed = float(cfg.experiment.lambda_fixed)
    adaptive = bool(cfg.experiment.adaptive_lambda)
    controller = (
        DivergenceController(
            float(cfg.experiment.lambda_target),
            float(cfg.experiment.lambda_eta),
            float(cfg.experiment.lambda_max),
        )
        if adaptive
        else None
    )

    best_val_bleu = -1.0
    best_epoch = -1
    div_best = float("nan")
    lam_best = lam_fixed
    trace_rows: List[Tuple[int, float, float, float]] = []
    patience_left = int(cfg.model.early_stop_patience)
    start = time.time()

    for epoch in range(int(cfg.model.epochs)):
        model.train()
        random.shuffle(train_pairs)
        probe = train_pairs[:batch_size]

        # Deterministic controller measurement (eval mode, no dropout).
        lam = lam_fixed
        div_t = float("nan")
        if use_reg:
            psrc, ptgt = make_batch(probe, pad, device)
            psrc_pad = make_pad_mask(psrc, pad)
            ptgt_pad = make_pad_mask(ptgt[:, :-1], pad)
            pcausal = make_causal_mask(ptgt.size(1) - 1, device)
            with torch.no_grad():
                model.eval()
                mats = model(
                    psrc, ptgt[:, :-1], psrc_pad, ptgt_pad, pcausal, collect_attn=True
                )["attn_mats"]
                div_t = float(transformer_attention_divergence(mats))
                model.train()
            if adaptive:
                lam = controller.update(div_t)

        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i : i + batch_size]
            src, tgt = make_batch(batch, pad, device)
            src_pad = make_pad_mask(src, pad)
            tgt_pad = make_pad_mask(tgt[:, :-1], pad)
            causal = make_causal_mask(tgt.size(1) - 1, device)
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1], src_pad, tgt_pad, causal,
                        collect_attn=(use_reg and i == 0))
            loss = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)),
                tgt[:, 1:].reshape(-1),
                ignore_index=pad,
            )
            if use_reg and i == 0:
                loss = loss + lam * transformer_attention_divergence(out["attn_mats"])
            loss.backward()
            optimizer.step()

        val_bleu = _bleu(model, data["val"], data, device, val_subset, sp)
        if adaptive:
            trace_rows.append((epoch, div_t, lam, val_bleu))
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            best_epoch = epoch
            lam_best = lam
            div_best = div_t
            patience_left = int(cfg.model.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"Early stop at epoch {epoch}")
                break

    if adaptive and trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "div_t", "lambda_t", "val_acc"])
            writer.writerows(trace_rows)

    test_bleu = _bleu(model, data["test"], data, device, len(data["test"]), sp)
    logger.info(f"val BLEU {best_val_bleu:.2f} @epoch {best_epoch}, test BLEU {test_bleu:.2f}")

    return {
        "best_epoch": float(best_epoch),
        "epochs_run": float(epoch + 1),
        "val_acc": best_val_bleu,
        "test_acc": test_bleu,
        "params": float(sum(p.numel() for p in model.parameters())),
        "wall_time_s": time.time() - start,
        "lambda_final": lam_best,
        "div_final": div_best,
    }
