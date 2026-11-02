"""Parse generated JSON and score it against the gold entities.

An entity counts as correct when its (entity text, type) pair matches a gold pair of the same sentence.
Pairs are compared as multisets, so a span that occurs twice in a sentence must be generated twice.
We also report a type-agnostic score ("span F1") to separate boundary errors from type errors.
"""
import json
import re
from collections import Counter

_OBJ = re.compile(r"\{[^{}]*\}")


def _clean(obj):
    out = []
    if isinstance(obj, list):
        for o in obj:
            if isinstance(o, dict) and isinstance(o.get("entity"), str) and isinstance(o.get("type"), str):
                out.append((o["entity"].strip(), o["type"].strip()))
    return out


def parse_prediction(text):
    """-> (list of (entity, type), is_valid_json). A fully valid answer is one JSON list of well-formed objects.
    Broken output is salvaged object by object so that one malformed entry does not zero the sentence."""
    text = text.strip()
    try:
        obj = json.loads(text)
        pairs = _clean(obj)
        return pairs, isinstance(obj, list) and len(pairs) == len(obj)
    except json.JSONDecodeError:
        pass
    pairs = []
    for m in _OBJ.finditer(text):
        try:
            pairs += _clean([json.loads(m.group(0))])
        except json.JSONDecodeError:
            continue
    return pairs, False


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(100 * p, 2), round(100 * r, 2), round(100 * (2 * p * r / (p + r) if p + r else 0.0), 2)


def score(pred_texts, gold_rows):
    """pred_texts: raw generations; gold_rows: dataset rows with "entities"."""
    tp = fp = fn = stp = sfp = sfn = valid = 0
    for text, row in zip(pred_texts, gold_rows):
        pred, ok = parse_prediction(text)
        valid += ok
        p, g = Counter(pred), Counter((e["entity"], e["type"]) for e in row["entities"])
        hit = sum((p & g).values())
        tp, fp, fn = tp + hit, fp + sum(p.values()) - hit, fn + sum(g.values()) - hit
        ps, gs = Counter(e for e, _ in pred), Counter(e["entity"] for e in row["entities"])
        hit = sum((ps & gs).values())
        stp, sfp, sfn = stp + hit, sfp + sum(ps.values()) - hit, sfn + sum(gs.values()) - hit
    precision, recall, f1 = _prf(tp, fp, fn)
    _, _, span_f1 = _prf(stp, sfp, sfn)
    return {"precision": precision, "recall": recall, "f1": f1, "span_f1": span_f1,
            "valid_json": round(100 * valid / max(len(gold_rows), 1), 2)}
