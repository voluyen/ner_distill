"""En-ViMedNER (English side) as a text -> JSON generation task for causal LMs.

Source: https://huggingface.co/datasets/nhuvo/En-ViMedNER (UMLS ST21pv, 21 semantic types).
Each sentence is one Label Studio-style JSON file; we only use `data.en_text` and the `en_label` spans.

Input  : the sentence.
Output : [{"entity": "<exact span text>", "type": "<UMLS semantic type name>"}, ...]  (sorted by position)

The model is trained with plain next-token prediction on prompt + JSON + EOS, with the loss on the JSON only.

    python -m nerllm.data                       # download + convert train/dev/test -> cache/envimed/*.jsonl
    python -m nerllm.data --splits mini-test    # just the 450-sentence mini-test
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import torch

REPO = "nhuvo/En-ViMedNER"
SPLITS = ("train", "dev", "test", "mini-test")
CACHE_DIR = "cache/envimed"

# MedMentions ST21pv semantic types (UMLS TUI -> name)
UMLS_TYPES = {
    "T005": "Virus",
    "T007": "Bacterium",
    "T017": "Anatomical Structure",
    "T022": "Body System",
    "T031": "Body Substance",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T058": "Health Care Activity",
    "T062": "Research Activity",
    "T074": "Medical Device",
    "T082": "Spatial Concept",
    "T091": "Biomedical Occupation or Discipline",
    "T092": "Organization",
    "T097": "Professional or Occupational Group",
    "T098": "Population Group",
    "T103": "Chemical",
    "T168": "Food",
    "T170": "Intellectual Product",
    "T201": "Clinical Attribute",
    "T204": "Eukaryote",
}

INSTRUCTION = ('Extract all biomedical named entities from the text. Answer with a JSON list of objects '
               'with the keys "entity" (the exact span from the text) and "type".')


def build_prompt(text, types_in_prompt=False, shots=()):
    """Plain-text task prompt. `shots` = rows (text + entities) shown as worked examples before the query."""
    head = INSTRUCTION
    if types_in_prompt:
        head += "\nEntity types: " + "; ".join(UMLS_TYPES.values()) + "."
    ex = "".join(f"Text: {r['text']}\nEntities:\n{entities_to_json(r['entities'])}\n\n" for r in shots)
    return f"{head}\n\n{ex}Text: {text}\nEntities:\n"


def use_chat(tok, chat=True):
    return bool(chat and getattr(tok, "chat_template", None))


def render(tok, user_text, chat=True):
    """Prompt string given to the tokenizer: wrapped in the model's chat template (instruction-tuned models such as
    MedGemma) so that generation starts right after the "model" turn header, or the raw text for base models."""
    if use_chat(tok, chat):
        return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                       add_generation_prompt=True)
    return user_text


def end_ids(tok, chat=True):
    """Token(s) that end an answer: <end_of_turn> (Gemma chat) plus EOS."""
    ids = [tok.eos_token_id]
    if use_chat(tok, chat):
        e = tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(e, int) and e != tok.unk_token_id and e not in ids:
            ids.insert(0, e)
    return ids


def pick_shots(rows, k):
    """k fixed, short, non-trivial training sentences that together cover as many entity types as possible."""
    if k <= 0:
        return []
    cand = [r for r in rows if 2 <= len(r["entities"]) <= 5 and len(r["text"]) <= 300]
    out, seen = [], set()
    for _ in range(k):
        best = max((r for r in cand if r not in out), key=lambda r: len({e["type"] for e in r["entities"]} - seen),
                   default=None)  # first max wins -> deterministic
        if best is None:
            break
        out.append(best)
        seen |= {e["type"] for e in best["entities"]}
    return out


def entities_to_json(entities):
    """Target string. `entities` = [{"entity": str, "type": str}, ...]."""
    return json.dumps([{"entity": e["entity"], "type": e["type"]} for e in entities], ensure_ascii=False)


# ---------------------------------------------------------------- download / conversion
def download(splits, cache_dir=CACHE_DIR):
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, repo_type="dataset", local_dir=os.path.join(cache_dir, "raw"),
                             allow_patterns=[f"{s}/*" for s in splits])


def parse_file(path, stats):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    text = raw["data"]["en_text"]
    ents = []
    for ann in raw["annotations"][0]["result"]:
        if ann.get("from_name") != "en_label":
            continue  # skip the Vietnamese side
        v = ann["value"]
        if text[v["start"]:v["end"]] != v["text"]:
            stats["offset_mismatch"] += 1
        if len(v["labels"]) > 1:
            stats["multi_label"] += 1
        code = v["labels"][0]
        if code not in UMLS_TYPES:
            stats["unknown_type"] += 1
            continue
        # the span is read from the sentence, so the target is always an exact substring
        ents.append({"entity": text[v["start"]:v["end"]], "type": UMLS_TYPES[code], "code": code,
                     "start": v["start"], "end": v["end"]})
    ents.sort(key=lambda e: (e["start"], e["end"]))
    return {"id": os.path.splitext(os.path.basename(path))[0], "text": text, "entities": ents}


def convert_split(split, cache_dir=CACHE_DIR):
    """raw/<split>/**.json -> <cache_dir>/<split>.jsonl (one sentence per line)."""
    root = os.path.join(cache_dir, "raw", split)
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"no sentence files under {root}: run `python -m nerllm.data --splits {split}`")
    stats = Counter()
    rows = [parse_file(p, stats) for p in files]
    rows = [r for r in rows if r["text"].strip()]
    out = os.path.join(cache_dir, f"{split}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    types = Counter(e["type"] for r in rows for e in r["entities"])
    print(f"{split}: {len(rows)} sentences, {sum(types.values())} entities, "
          f"{sum(1 for r in rows if not r['entities'])} without entities | {dict(stats)} -> {out}")
    return rows


def load_split(split, cache_dir=CACHE_DIR, num=-1):
    path = os.path.join(cache_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python -m nerllm.data --splits {split}` first")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows if num < 0 else rows[:num]


# ---------------------------------------------------------------- character spans for MTA
def compute_char_spans(texts, cache_dir=None):
    """Per sentence: (word_spans, phrase_spans) as lists of (char_start, char_end).

    Words = spaCy tokens; phrases = noun chunks + verb phrases (non-overlapping), as in nerkd. Cached by hash.
    """
    key = hashlib.md5(json.dumps(texts).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"charspans_{key}.json") if cache_dir else None
    if path and os.path.exists(path):
        return json.load(open(path))
    import spacy
    from spacy.matcher import Matcher
    from nerkd.data import VERB_PHRASE, _non_overlapping
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASE", [VERB_PHRASE])
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        words = [(t.idx, t.idx + len(t)) for t in doc if not t.is_space]
        ph = [(s, e) for _, s, e in matcher(doc)] + [(c.start, c.end) for c in doc.noun_chunks]
        ph = [(doc[s].idx, doc[e - 1].idx + len(doc[e - 1])) for s, e in _non_overlapping(ph)]
        out.append((words, ph))
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(out, open(path, "w"))
    return out


def span_map(offsets, base, n_chars, spans):
    """Token -> span id (-1 outside). `offsets` = char offsets of the tokens in the rendered prompt, `base` = where
    the sentence starts in that string. A token takes the span of its first character that lies in a span."""
    c2s = [-1] * n_chars
    for i, (s, e) in enumerate(spans):
        c2s[s:e] = [i] * (e - s)
    out = []
    for s, e in offsets:
        ids = [c2s[c - base] for c in range(max(s, base), min(e, base + n_chars)) if c2s[c - base] >= 0]
        out.append(ids[0] if ids else -1)
    return out


# ---------------------------------------------------------------- tokenization for next-token SFT
class SFTDataset(torch.utils.data.Dataset):
    """input_ids = prompt + target + end token; labels = -100 on the prompt.

    Prompt and target are tokenized separately and concatenated, so the target always starts at
    a clean token boundary and training matches generation (which starts from the tokenized prompt).
    With `teacher_shots` the teacher gets its own few-shot prompt in front of the *same* target tokens
    (t_input_ids / t_labels); the supervised positions are identical, so the logits line up one to one.
    """

    def __init__(self, rows, tokenizer, max_length=768, types_in_prompt=False, chat=True, teacher_shots=(),
                 with_spans=False, cache_dir=None):
        self.items, self.dropped = [], 0
        end = end_ids(tokenizer, chat)[0]
        enc = lambda txt: tokenizer(render(tokenizer, txt, chat), add_special_tokens=False)["input_ids"]
        spans = compute_char_spans([r["text"] for r in rows], cache_dir) if with_spans else None

        def maps(user_text, r, i, n_target):  # word / phrase span maps over prompt + target
            rendered = render(tokenizer, user_text, chat)
            base = rendered.rindex("Text: " + r["text"] + "\nEntities:") + len("Text: ")
            offs = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            return [span_map(offs, base, len(r["text"]), sp) + [-1] * n_target for sp in spans[i]]

        for i, r in enumerate(rows):
            p = enc(build_prompt(r["text"], types_in_prompt))
            t = tokenizer(entities_to_json(r["entities"]), add_special_tokens=False)["input_ids"] + [end]
            if len(p) + len(t) > max_length:  # truncating the target would teach invalid JSON
                self.dropped += 1
                continue
            item = {"input_ids": p + t, "labels": [-100] * len(p) + t}
            if with_spans:
                item["word_span"], item["phrase_span"] = maps(build_prompt(r["text"], types_in_prompt), r, i, len(t))
            if teacher_shots:
                tp = enc(build_prompt(r["text"], types_in_prompt, teacher_shots))
                item.update(t_input_ids=tp + t, t_labels=[-100] * len(tp) + t)
                if with_spans:
                    item["t_word_span"], item["t_phrase_span"] = maps(
                        build_prompt(r["text"], types_in_prompt, teacher_shots), r, i, len(t))
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def pad(seqs, value):
        L = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (L - len(s)) for s in seqs], dtype=torch.long)

    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        out = {"input_ids": pad(ids, pad_id), "attention_mask": pad([[1] * len(i) for i in ids], 0),
               "labels": pad([b["labels"] for b in batch], -100)}
        if "t_input_ids" in batch[0]:
            t = [b["t_input_ids"] for b in batch]
            out.update(t_input_ids=pad(t, pad_id), t_attention_mask=pad([[1] * len(i) for i in t], 0),
                       t_labels=pad([b["t_labels"] for b in batch], -100))
        for k in ("word_span", "phrase_span", "t_word_span", "t_phrase_span"):
            if k in batch[0]:
                out[k] = pad([b[k] for b in batch], -1)
        return out
    return collate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=SPLITS)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    a = ap.parse_args()
    download(a.splits, a.cache_dir)
    for s in a.splits:
        convert_split(s, a.cache_dir)
