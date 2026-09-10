"""Multi30k (En->De) loading with a shared sentencepiece tokenizer.

Downloads the raw task1 files from the official GitHub mirror (no torchtext needed)
and trains an 8k shared BPE model.
"""

import gzip
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
FILES = {
    "train": ("train.en.gz", "train.de.gz"),
    "val": ("val.en.gz", "val.de.gz"),
    "test": ("test_2016_flickr.en.gz", "test_2016_flickr.de.gz"),
}


def _download(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Fallback chain: mirror prefixes x connection routes.
    # - gh-proxy.com: verified working in restricted networks (no proxy needed)
    # - raw GitHub: works on Colab and open networks
    # - Clash local proxy (7890): last resort for the author's machine
    mirrors = ["https://gh-proxy.com/", ""]
    openers = [
        urllib.request.build_opener(urllib.request.ProxyHandler({})),
        urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
            )
        ),
    ]
    for pair in FILES.values():
        for name in pair:
            target = raw / name
            if target.exists():
                continue
            logger.info(f"downloading {name}")
            last_err: Exception | None = None
            data = None
            for mirror in mirrors:
                url = mirror + BASE_URL + name
                for opener in openers:
                    try:
                        with opener.open(url, timeout=30) as resp:
                            data = resp.read()
                        break
                    except Exception as e:  # noqa: BLE001 - fallback chain, last error re-raised
                        last_err = e
                if data is not None:
                    break
            if data is None:
                raise RuntimeError(f"failed to download {name}: {last_err}")
            with gzip.open(target, "wb") as f:
                f.write(data)


def _read_gz_lines(path: Path) -> List[str]:
    data = path.read_bytes()
    # Some mirrors re-gzip already-gzipped files; decompress while the gzip
    # magic (1f 8b) is present, up to a sane bound.
    for _ in range(3):
        if data[:2] != b"\x1f\x8b":
            break
        data = gzip.decompress(data)
    return [line.strip() for line in data.decode("utf-8").splitlines()]


def _train_tokenizer(root: Path, vocab_size: int) -> None:
    spm_path = root / f"spm_{vocab_size}.model"
    if spm_path.exists():
        return
    raw = root / "raw"
    corpus = root / "corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for en, de in FILES.values():
            for name in (en, de):
                for line in _read_gz_lines(raw / name):
                    f.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / f"spm_{vocab_size}"),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        # Standard layout with distinct ids: bos_id == eos_id triggers a
        # sentencepiece internal insert_id conflict, so ids are kept separate.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )


def load_multi30k(root: str = "data", vocab_size: int = 8000) -> Dict:
    """Load Multi30k and return tokenized splits plus vocab metadata.

    Returns:
        dict with keys: src_vocab, tgt_vocab, pad_idx, bos_id, train, val, test.
        Each split is a list of (src_ids: List[int], tgt_ids: List[int]) where
        src ends with EOS and tgt starts with BOS and ends with EOS.
    """
    root = Path(root)
    _download(root)
    _train_tokenizer(root, vocab_size)
    sp = spm.SentencePieceProcessor(model_file=str(root / f"spm_{vocab_size}.model"))

    def tokenize(text: str, bos: bool) -> List[int]:
        ids = sp.EncodeAsIds(text)
        return ([sp.bos_id()] if bos else []) + ids + [sp.eos_id()]

    splits: Dict[str, List[Tuple[List[int], List[int]]]] = {}
    for split, (en, de) in FILES.items():
        pairs = zip(_read_gz_lines(root / "raw" / en), _read_gz_lines(root / "raw" / de))
        splits[split] = [
            (tokenize(s, bos=False), tokenize(t, bos=True)) for s, t in pairs
        ]

    return {
        "src_vocab": vocab_size,
        "tgt_vocab": vocab_size,
        "pad_idx": sp.pad_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "sp": sp,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }


def make_batch(
    pairs: List[Tuple[List[int], List[int]]],
    pad_idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (src, tgt) pairs into (B, T_s) and (B, T_t) tensors."""
    srcs = [torch.tensor(s, device=device) for s, _ in pairs]
    tgts = [torch.tensor(t, device=device) for _, t in pairs]
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    return src, tgt
