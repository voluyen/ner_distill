"""CoNLL-2003 loading + tokenization for encoder -> encoder NER distillation.

Teacher and student share one tokenizer, so a batch is tokenized once and both models see the
same input_ids. Labels sit on the first sub-token of each word (`label_pos="first"`), or on the last
one for causal decoders (`"last"`, so the labelled position has seen the whole word); other
sub-tokens and special tokens get -100 and are ignored by CE, KD and evaluation.

Byte-level BPE tokenizers (Qwen, GPT-2 style) drop the leading space when fed pre-split words, which
turns " founded" into "f" + "ounded"; for those the sentence is tokenized as plain text and tokens are
mapped back to words through character offsets.

For MTA, every sentence also gets word spans (each word) and phrase spans (spaCy noun chunks and
verb phrases over the gold word segmentation), stored as token -> span-id maps.
"""
import hashlib
import json
import os

import torch
from datasets import load_dataset

VERB_PHRASE = [{"POS": "AUX", "OP": "*"}, {"POS": "ADV", "OP": "*"},
               {"POS": "VERB", "OP": "+"}, {"POS": "ADV", "OP": "*"}]


def load_conll(dataset_name="eriktks/conll2003"):
    # datasets>=3 no longer runs loading scripts; use the Hub's auto-converted parquet branch
    try:
        ds = load_dataset(dataset_name)
    except RuntimeError:
        ds = load_dataset(dataset_name, revision="refs/convert/parquet")
    labels = ds["train"].features["ner_tags"].feature.names
    return ds, labels


def _needs_join(tokenizer):
    """True if tokenizing pre-split words differs from tokenizing the sentence (byte-level BPE)."""
    if not hasattr(tokenizer, "_nerkd_join"):
        words = ["Steve", "Jobs", "founded", "Apple"]
        split = tokenizer(words, is_split_into_words=True)["input_ids"]
        tokenizer._nerkd_join = split != tokenizer(" ".join(words))["input_ids"]
    return tokenizer._nerkd_join


def encode_words(tokenizer, words, max_length):
    """-> (input_ids, word_ids) with word_ids[i] = word index of token i, None for special tokens."""
    if not _needs_join(tokenizer):
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
        return enc["input_ids"], enc.word_ids()
    text, char2word = "", []
    for i, w in enumerate(words):
        if i:
            text += " "
            char2word.append(None)
        text += w
        char2word += [i] * len(w)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    word_ids = []
    for s, e in enc["offset_mapping"]:
        ws = [char2word[c] for c in range(s, e) if char2word[c] is not None]
        word_ids.append(ws[0] if ws else None)
    return enc["input_ids"], word_ids


def label_mask(word_ids, label_pos="first"):
    """1 on the token that carries each word's label (first or last sub-token), else 0."""
    n = len(word_ids)
    if label_pos == "first":
        return [int(w is not None and (i == 0 or word_ids[i - 1] != w)) for i, w in enumerate(word_ids)]
    return [int(w is not None and (i == n - 1 or word_ids[i + 1] != w)) for i, w in enumerate(word_ids)]


def _non_overlapping(spans):
    """Keep the longest span at each start, drop spans overlapping an already kept one."""
    kept, last_end = [], 0
    for s, e in sorted(spans, key=lambda x: (x[0], -x[1])):
        if s >= last_end:
            kept.append((s, e))
            last_end = e
    return kept


def compute_spans(sentences, cache_dir=None):
    """Phrase spans (word start, word end) per sentence: noun chunks + verb phrases via spaCy.

    spaCy runs on the gold words (no re-tokenization), so span boundaries are word indices.
    Results are cached as JSON keyed by the sentences' hash.
    """
    key = hashlib.md5(json.dumps(sentences).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"spans_{key}.json") if cache_dir else None
    if path and os.path.exists(path):
        return json.load(open(path))

    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASE", [VERB_PHRASE])
    docs = (Doc(nlp.vocab, words=words) if words else Doc(nlp.vocab, words=["."]) for words in sentences)
    out = []
    for doc in nlp.pipe(docs, batch_size=256):
        spans = [(start, end) for _, start, end in matcher(doc)]
        spans += [(nc.start, nc.end) for nc in doc.noun_chunks]
        out.append(_non_overlapping(spans))
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(out, open(path, "w"))
    return out


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, max_length=256, with_spans=False, cache_dir=None, label_pos="first"):
        split = [ex for ex in split if len(ex["tokens"]) > 0]
        phrases = compute_spans([ex["tokens"] for ex in split], cache_dir) if with_spans else None
        self.items = []
        for i, ex in enumerate(split):
            input_ids, word_ids = encode_words(tokenizer, ex["tokens"], max_length)
            labels = [ex["ner_tags"][w] if m else -100 for w, m in zip(word_ids, label_mask(word_ids, label_pos))]
            item = {"input_ids": input_ids, "labels": labels}
            if with_spans:
                word2phrase = [-1] * len(ex["tokens"])
                for pid, (s, e) in enumerate(phrases[i]):
                    word2phrase[s:e] = [pid] * (e - s)
                item["word_span"] = [-1 if w is None else w for w in word_ids]
                item["phrase_span"] = [-1 if w is None else word2phrase[w] for w in word_ids]
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
        out = {
            "input_ids": pad([b["input_ids"] for b in batch], pad_id),
            "attention_mask": pad([[1] * len(b["input_ids"]) for b in batch], 0),
            "labels": pad([b["labels"] for b in batch], -100),
        }
        for k in ("word_span", "phrase_span"):
            if k in batch[0]:
                out[k] = pad([b[k] for b in batch], -1)
        return out
    return collate
