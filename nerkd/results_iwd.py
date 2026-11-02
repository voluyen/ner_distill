"""Mean ± std over seeds of CoNLL test F1 and OOD F1 (WNUT-17, OntoNotes) for results_iwd/.

Usage: python -m nerkd.results_iwd [results_iwd]
Runs `<name>_s<seed>` are grouped by `<name>`; std is the sample std (ddof=1) over seeds.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results_iwd"
groups = defaultdict(list)
for path in sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True)):
    rel = os.path.relpath(os.path.dirname(path), root)
    groups[re.sub(r"_s\d+$", "", rel)].append(json.load(open(path)))


def fmt(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"


cols = [("CoNLL dev", lambda r: r["dev"]["f1"]), ("CoNLL test", lambda r: r["test"]["f1"]),
        ("WNUT-17", lambda r: r["ood"]["wnut17"]["f1"]), ("OntoNotes", lambda r: r["ood"]["ontonotes"]["f1"])]
print("| run | seeds | params (M) | " + " | ".join(c for c, _ in cols) + " |")
print("|---|---|---|" + "---|" * len(cols))
for name in sorted(groups, key=lambda n: (n.split("/")[0], "sft" not in n, n)):
    rs = groups[name]
    vals = []
    for _, get in cols:
        try:
            vals.append(fmt([get(r) for r in rs]))
        except KeyError:
            vals.append("-")
    print(f"| {name} | {len(rs)} | {rs[0].get('params_M', '-')} | " + " | ".join(vals) + " |")
