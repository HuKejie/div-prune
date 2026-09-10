"""Fetch BibTeX for verified references into refs.bib (one-off utility).

arXiv entries via https://arxiv.org/bibtex/{id}; DOI entries via CrossRef
transform endpoint. Entry keys are rewritten to the paper's citation keys.
Review fixes (2026-08-21): validate BibTeX on every entry, isolate per-entry
failures, retry transient network errors, tolerant decoding, proper UA.
"""

import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ARXIV_IDS: dict[str, str] = {
    "michel2019": "1905.10650",
    "voita2019": "1905.09418",
    "chai2024": "2403.08058",
    "dha2024": "2406.06567",
    "proxyattn2025": "2509.24745",
    "chen2022": "2203.06345",
    "nicolicioiu2023": "2308.16274",
    "jhareagen2025": "2501.03489",
    "kendall2018": "1705.07115",
    "gradnorm2018": "1711.02257",
    "excessmtl2024": "2402.02009",
    "harvey2024": "2410.19675",
    "hinton2015": "1503.02531",
    "tinybert2020": "1909.10351",
    "minilm2020": "2002.10957",
    "glnn2022": "2110.08727",
    "krd2023": "2306.05628",
    "parkno2021": "2109.14960",
    "epsd2024": "2402.00084",
    "velickovic2018": "1710.10903",
    "yang2016": "1603.08861",
    "elliott2016": "1605.00459",
    "post2018": "1804.08771",
}

DOIS: dict[str, str] = {
    "yun2021": "10.3390/app11041548",
    "deacon2024": "10.3390/make6040126",
    "ddk2024": "10.1016/j.neunet.2024.106164",
}

# audhkhasi2022 has no DOI; ISCA archive only — hand-written entry below.
AUDHKHASI_MANUAL = """\
@inproceedings{audhkhasi2022,
  author    = {Kartik Audhkhasi and Andrew Rosenberg and Bhuvana Ramabhadran},
  title     = {Analysis of Self-Attention Head Diversity for Conformer-based Automatic Speech Recognition},
  booktitle = {Proc. Interspeech 2022},
  year      = {2022},
  publisher = {ISCA},
  doi       = {10.21437/Interspeech.2022-10698},
}
"""

# CrossRef BibTeX may start with leading whitespace; capture it so it is preserved.
ENTRY_KEY_RE = re.compile(r"^(\s*@\w+\{)[^,]+", re.MULTILINE)
# Extract just the citation key name for duplicate detection.
KEY_NAME_RE = re.compile(r"@\w+\{([^,]+)")
USER_AGENT = "paper-fetch/1.0 (div-prune citation utility)"
MAX_RETRIES = 3


def fetch(url: str, headers: dict | None = None) -> str:
    """Fetch a URL with retries; returns the decoded text body."""
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    last_exc: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"fetch failed after {MAX_RETRIES} retries: {url}") from last_exc


def rename_key(bibtex: str, new_key: str) -> str:
    """Rename the single @entry key; raises if the body has no/extra entries."""
    renamed, n = ENTRY_KEY_RE.subn(rf"\g<1>{new_key}", bibtex, count=0)
    if n != 1:
        preview = bibtex[:120].replace("\n", " ")
        raise ValueError(f"expected exactly 1 @entry, found {n}: {preview!r}")
    return renamed


def main() -> None:
    out = Path(__file__).parent / "refs.bib"
    entries: list[str] = []
    failures: list[str] = []

    for key, arxiv_id in ARXIV_IDS.items():
        try:
            raw = fetch(f"https://arxiv.org/bibtex/{arxiv_id}")
            entries.append(rename_key(raw, key))
        except (ValueError, RuntimeError) as exc:
            failures.append(f"{key} (arXiv:{arxiv_id}): {exc}")
        time.sleep(0.5)

    for key, doi in DOIS.items():
        try:
            url = (
                "https://api.crossref.org/works/"
                f"{urllib.parse.quote(doi, safe='')}/transform/application/x-bibtex"
            )
            raw = fetch(url, headers={"Accept": "application/x-bibtex"})
            entries.append(rename_key(raw, key))
        except (ValueError, RuntimeError) as exc:
            failures.append(f"{key} (DOI:{doi}): {exc}")
        time.sleep(0.5)

    entries.append(AUDHKHASI_MANUAL)

    keys = [m.group(1) for e in entries if (m := KEY_NAME_RE.search(e))]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        failures.append(f"duplicate entry keys: {sorted(duplicates)}")

    header = (
        f"% refs.bib — generated {date.today().isoformat()} by paper/fetch_bibtex.py\n"
        "% arXiv entries from arxiv.org/bibtex; DOI entries from CrossRef.\n"
        "% All identifiers verified against the arXiv API / Crossref "
        "(plan/literature-survey.md, 2026-08-16).\n\n"
    )
    out.write_text(header + "\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {out} with {len(entries)} entries.")
    if failures:
        print("FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
