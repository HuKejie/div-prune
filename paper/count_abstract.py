"""Count words in the abstract environment of main.tex (local counter)."""
import re
from pathlib import Path

tex = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
body = m.group(1)
body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", body)  # strip commands
words = body.split()
print("abstract words:", len(words))
