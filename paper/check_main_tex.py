"""Structural checks for main.tex (no TeX distribution on this machine).

Verifies: begin/end environment balance, \\cite keys exist in refs.bib,
\\label keys are unique, includegraphics/input/bibliography paths exist,
and no unescaped & in prose lines (valid inside tabular/usepackage).
Run from repo root: python paper/check_main_tex.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "refs.bib"


def _strip_comments(text: str) -> str:
    """Remove comment lines (and trailing comments) so they cannot desync checks."""
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def main() -> None:
    raw = TEX.read_text(encoding="utf-8")
    text = _strip_comments(raw)
    errors: list[str] = []

    # 1. environment balance (starred variants kept distinct)
    begins = re.findall(r"\\begin\{(\w+\*?)\}", text)
    ends = re.findall(r"\\end\{(\w+\*?)\}", text)
    for env in sorted(set(begins) | set(ends)):
        if begins.count(env) != ends.count(env):
            errors.append(f"env {env}: begin={begins.count(env)} end={ends.count(env)}")

    # 2. cite keys exist in refs.bib (incl. \citep/\citet and optional args;
    #    \nocite{*} skipped)
    bib_text = BIB.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+)", bib_text))
    cite_keys = set(re.findall(r"\\cite\w*(\[[^\]]*\])?\{([^}]+)\}", text))
    cited: set[str] = set()
    for _, c in cite_keys:
        for k in c.split(","):
            k = k.strip()
            if k and k != "*":
                cited.add(k)
    for key in sorted(cited - bib_keys):
        errors.append(f"cite key not in refs.bib: {key}")

    # 3. unique labels
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    for lab in set(labels):
        if labels.count(lab) > 1:
            errors.append(f"duplicate label: {lab}")

    # 4. referenced files exist
    for m in re.findall(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", text):
        p = ROOT / "paper" / m[1]
        if not p.exists():
            errors.append(f"missing figure: {m[1]}")
    for m in re.findall(r"\\input\{([^}]+)\}", text):
        p = ROOT / "paper" / m
        if not p.suffix:
            p = p.with_suffix(".tex")
        if not p.exists():
            errors.append(f"missing input file: {m}")
    for m in re.findall(r"\\bibliography\{([^}]+)\}", text):
        p = (ROOT / "paper" / m).with_suffix(".bib")
        if not p.exists():
            errors.append(f"missing bibliography: {m}")

    # 5. unescaped & in prose (valid inside tabular and usepackage options)
    in_tabular = False
    for i, line in enumerate(raw.splitlines(), 1):
        s = line.split("%", 1)[0].strip()
        if "\\begin{tabular" in s:
            in_tabular = True
        if "\\end{tabular" in s:
            in_tabular = False
        if not s or in_tabular or "\\usepackage" in s:
            continue
        if "&" in s:
            errors.append(f"line {i}: unescaped & in {s[:60]!r}")

    if errors:
        print("CHECKS FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"CHECKS PASSED: {len(begins)} environments balanced, "
          f"{len(cited)} cite keys verified, {len(labels)} unique labels.")


if __name__ == "__main__":
    main()
