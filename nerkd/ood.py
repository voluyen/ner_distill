"""Out-of-distribution evaluation of a CoNLL-2003 model on WNUT-17 and OntoNotes 5.0 (IWD protocol).

The model predicts CoNLL labels (PER/ORG/LOC/MISC). Gold types of each OOD dataset are mapped to
PER/ORG/LOC; the rest are either dropped or ignored:

  * mapped  -> scored as that CoNLL type
  * ignored -> the span is removed from gold, and predicted spans overlapping it are removed too
               (the model is neither rewarded nor punished there, e.g. NORP ~ MISC, product, FAC)
  * absent  -> the type is not an entity in CoNLL terms (DATE, MONEY, ...): gold becomes O, so a
               PER/ORG/LOC prediction there counts as a false positive

Predicted MISC spans are always dropped: none of the OOD schemas has a clean MISC counterpart.
Score = entity-level micro P/R/F1 with exact span + type match (spans from seqeval's conlleval-style
`get_entities`, same chunking as `seqeval.metrics.f1_score`).

IWD (arXiv 2609.22566) does not publish its label mapping; this one is ours.
"""
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import DataLoader

from nerkd.data import encode_words, label_mask

OOD = {
    "wnut17": {
        "hf": "tner/wnut2017",
        "map": {"person": "PER", "location": "LOC", "corporation": "ORG", "group": "ORG"},
        "ignore": {"product", "creative-work"},
    },
    "ontonotes": {
        "hf": "tner/ontonotes5",
        "map": {"PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC"},
        "ignore": {"NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PRODUCT"},
    },
}


def load_ood(name, split="test"):
    """-> list of (tokens, gold entities [(type, start, end)], ignored spans [(start, end)])."""
    import json
    cfg = OOD[name]
    label2id = json.load(open(hf_hub_download(cfg["hf"], "dataset/label.json", repo_type="dataset")))
    id2label = {i: l for l, i in label2id.items()}
    ds = load_dataset(cfg["hf"], revision="refs/convert/parquet")[split]
    out = []
    for ex in ds:
        if not ex["tokens"]:
            continue
        gold, ignored = [], []
        for typ, s, e in get_entities([id2label[t] for t in ex["tags"]]):
            if typ in cfg["map"]:
                gold.append((cfg["map"][typ], s, e))
            elif typ in cfg["ignore"]:
                ignored.append((s, e))
        out.append((ex["tokens"], gold, ignored))
    return out


class _Words(torch.utils.data.Dataset):
    def __init__(self, sents, tokenizer, max_length, label_pos):
        self.items = []
        for tokens, _, _ in sents:
            input_ids, word_ids = encode_words(tokenizer, tokens, max_length)
            self.items.append({"input_ids": input_ids, "labels": label_mask(word_ids, label_pos)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def score(sents, pred_tags):
    tp = n_pred = n_gold = 0
    for (tokens, gold, ignored), tags in zip(sents, pred_tags):
        tags = tags + ["O"] * (len(tokens) - len(tags))  # words cut by truncation -> O
        pred = [(t, s, e) for t, s, e in get_entities(tags)
                if t != "MISC" and not any(s <= ie and is_ <= e for is_, ie in ignored)]
        gold_set = set(gold)
        tp += sum(p in gold_set for p in set(pred))
        n_pred += len(set(pred))
        n_gold += len(gold_set)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f, 2)}


@torch.no_grad()
def evaluate_ood(model, tokenizer, id2label, collate, device, amp_dtype, names=("wnut17", "ontonotes"),
                 max_length=256, batch_size=128, label_pos="first", cache={}):
    model.eval()
    out = {}
    for name in names:
        if name not in cache:
            cache[name] = load_ood(name)
        sents = cache[name]
        loader = DataLoader(_Words(sents, tokenizer, max_length, label_pos), batch_size=batch_size, collate_fn=collate)
        pred_tags = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            for p_row, m_row in zip(logits.argmax(-1).cpu().tolist(), batch["labels"].cpu().tolist()):
                pred_tags.append([id2label[p] for p, m in zip(p_row, m_row) if m == 1])
        out[name] = score(sents, pred_tags)
    model.train()
    return out


if __name__ == "__main__":
    # evaluate saved checkpoints: python -m nerkd.ood <ckpt_dir> [...]; merges "ood" into results.json
    import json
    import os
    import sys

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from nerkd.data import make_collate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ckpt in sys.argv[1:]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForTokenClassification.from_pretrained(ckpt).to(device)
        res = evaluate_ood(model, tok, model.config.id2label, make_collate(tok.pad_token_id), device, torch.bfloat16,
                           label_pos=getattr(model.config, "nerkd_label_pos", "first"))
        print(ckpt, res, flush=True)
        path = os.path.join(ckpt, "results.json")
        r = json.load(open(path)) if os.path.exists(path) else {}
        r["ood"] = res
        json.dump(r, open(path, "w"), indent=2)
