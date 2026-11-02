"""Collect results.json of the MedGemma runs into one markdown table.

    python -m nerllm.results [results_llm]
"""
import json
import os
import sys

# (label, relative dir) in display order; entries without results.json are skipped
ROWS = [
    ("Teacher MedGemma-27B (5-shot, no training)", "zeroshot/google_medgemma-27b-it"),
    ("Student MedGemma-4B (5-shot, no training)", "zeroshot/google_medgemma-4b-it"),
    ("DistiLLM SKL", "kd/skl"),
    ("DistiLLM SRKL", "kd/srkl"),
    ("AMiD", "kd/amid"),
    ("MTA only", "kd/mta_only"),
    ("SRKL + MTA", "kd/srkl_mta"),
    ("AMiD + MTA", "kd/amid_mta"),
]


def main(root="results_llm"):
    out = ["| Method | Dev F1 | Test P | Test R | Test F1 | Span F1 | Valid JSON % | Params (M) |",
           "|---|---|---|---|---|---|---|---|"]
    for label, d in ROWS:
        path = os.path.join(root, d, "results.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        t = r["test"]
        out.append(f"| {label} | {r['dev']['f1']:.2f} | {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} "
                   f"| {t['span_f1']:.2f} | {t['valid_json']:.2f} | {r.get('params_M', '')} |")
    if len(out) == 2:
        out.append("| (no results found) |  |  |  |  |  |  |  |")
    text = "# MedGemma-27B -> MedGemma-4B on En-ViMedNER (generative NER)\n\n" + "\n".join(out) + "\n"
    print(text)
    return text


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results_llm"
    text = main(root)
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "RESULTS.md"), "w").write(text)
