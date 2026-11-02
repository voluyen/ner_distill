"""Print a markdown table of every results.json under a results directory.

Usage: python -m nerkd.results [results_dir]
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "results"
print("| run | params (M) | dev F1 | test P | test R | test F1 |\n|---|---|---|---|---|---|")
for path in sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True)):
    r = json.load(open(path))
    t = r["test"]
    name = os.path.relpath(os.path.dirname(path), root)
    print(f"| {name} | {r.get('params_M', '-')} | {r['dev']['f1']} | {t['precision']} | {t['recall']} | {t['f1']} |")
