import re

tex = open("paper/main.tex", encoding="utf-8").read()
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
body = m.group(1)
body = re.sub(r"%.*", "", body)
# strip commands but keep their arguments' words
words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'+-]*", body)
print("abstract words (approx, excl. math):", len(words))

orig = """Fixed-weight diversity penalties for multi-head attention have no usable
operating point: weights at or below 1 leave inter-head similarity unchanged,
while weights at or above 5 move similarity at the cost of 1.2--3.1 points of
accuracy. We introduce an adaptive controller that treats a target
head-similarity budget as a set point and adjusts the regularization weight
per epoch through an integral control law. On three citation networks (Cora,
Citeseer, Pubmed) with 2-layer GATs and 10 seeds per configuration, the
controller enforces the budget late in training while test accuracy remains
within $\\pm 0.4$ points of the unregularized baseline --- an effect no fixed
weight achieves. Reusing the same similarity metric as a head-pruning
criterion, pruning to 4 of 8 heads (50\\% of head parameters) followed by
knowledge distillation recovers $82.00 \\pm 0.69$ accuracy and matches students
trained from scratch with KD at identical budgets, while fine-tuning alone
leaves a 2.1--2.4 point gap. A paired analysis over pruning criteria
(similarity, gradient, random, magnitude) finds no reliable difference among
them at this scale --- fine-tuning dominates criterion choice --- and we report
this negative result with strict statistics. We do not claim that the
regularizer improves accuracy or that the diversity criterion outperforms
existing ones."""
orig = re.sub(r"%.*", "", orig)
print("original abstract (same method):", len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'+-]*", orig)))
