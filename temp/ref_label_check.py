import re
from pathlib import Path

ROOT = Path("paper")
tex = (ROOT / "main.tex").read_text(encoding="utf-8")
for p in sorted(ROOT.glob("tables/*.tex")):
    tex += p.read_text(encoding="utf-8")
tex = re.sub(r"%.*", "", tex)

refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
missing = refs - labels
unused = labels - refs
print("missing label targets:", sorted(missing) if missing else "none")
print("labels never referenced:", sorted(unused) if unused else "none")
print(f"total refs: {len(refs)}, total labels: {len(labels)}")
