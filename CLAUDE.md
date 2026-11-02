# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Knowledge distillation research code for NER, with two independent pipelines that share the loss code:

- **`nerkd/`** – encoder/decoder token classification on CoNLL-2003 (teacher → student, e.g. `bert-base-cased` → `distilbert-base-cased`, or Qwen2.5-7B → 0.5B under the IWD protocol). Distillation losses: Hinton KL variants (`fkl rkl sym js tv`), DistiLLM (`skl srkl`), AMiD (`amid`), and MTA span-alignment of hidden states (`--w-mta > 0`, combinable with any `--kd-type`).
- **`nerllm/`** – generative NER (text → JSON) on En-ViMedNER, MedGemma-27B (no SFT, few-shot teacher) → MedGemma-4B. Reuses `nerkd/losses.py`. See `nerllm/README.md`.

`README.md` has the full experiment tables, IWD label mapping and results; don't duplicate them here.

## Commands

Setup (`.venv` via uv, Python 3.12, torch 2.4.1 + transformers 4.42.4 pinned):
```bash
bash setup.sh
```

Encoder pipeline (all experiments are defined in the `EXPERIMENTS` array in `scripts/run.sh`; add a config by adding one line there):
```bash
bash scripts/run.sh list                 # list experiments
bash scripts/run.sh teacher              # teacher must be trained first; all KD runs load TEACHER_CKPT
bash scripts/run.sh amid_mta             # one experiment; extra args are forwarded to nerkd.train
bash scripts/run_all.sh [exp ...]        # sequential, skips runs with an existing results.json (FORCE=1 to rerun)
.venv/bin/python -m nerkd.results results   # markdown table from results/**/results.json
# quick debug run (single "test"):
SUFFIX=_debug bash scripts/run.sh amid_mta --train-num 800 --eval-num 300 --epochs 1
```
Env overrides: `GPU SEED SUFFIX TEACHER STUDENT TEACHER_CKPT RESULTS MTA_WORD MTA_PHRASE OOD` (`OOD=` disables OOD eval; default `wnut17 ontonotes`).

IWD protocol (multi-seed, parallel, CoNLL + OOD):
```bash
PAIRS=bert SEEDS="1 2" JOBS=4 bash scripts/run_iwd.sh amid   # PAIRS: qwen (default) | bert | deberta
python -m nerkd.ood <ckpt_dir> ...                            # OOD eval of an existing checkpoint
```
`run_iwd.sh` hard-codes per-pair model names, MTA layer pairs and extra train args in `pair_env()`.

Generative NER (needs a separate env: Gemma 3 requires transformers>=4.50, so `.venv-gemma` from `nerllm/requirements-gemma.txt`; models are gated on HF):
```bash
bash scripts/run_llm.sh prepare          # one-time data download → cache/envimed/*.jsonl
bash scripts/run_llm.sh list
SUFFIX=_debug bash scripts/run_llm.sh srkl --train-num 50 --dev-num 20 --test-num 20 --epochs 1   # smoke test
DRY=1 bash scripts/run_llm.sh srkl       # print the command only
export HF_TOKEN=... && bash run_medgemma.sh   # one-shot: env + access check + data + smoke test + all runs
```

There is no test suite or linter configured; validate changes with small `--train-num/--eval-num` debug runs. No git repo.

## Architecture notes

- **Experiment registry pattern**: `scripts/run.sh` and `scripts/run_llm.sh` are pipe-delimited `name | output dir | args` tables that `eval` a `python -m nerkd.train` / `nerllm.train` command. Outputs go to `results[_iwd|_llm]/<group>/<name><SUFFIX>/` (checkpoint, `log.txt`, `results.json`); logs to `logs/`. `results/`, `results_iwd/`, `results_llm/`, `logs/`, `cache/`, `third_party/` are git-ignored.
- **Shared tokenizer assumption (nerkd)**: teacher and student share a vocab, so each batch is tokenized once and fed to both. Causal models (Qwen) use `--label-pos last` (label on the last sub-token of each word) and `nerkd.data.encode_words` maps BPE tokens to words by char offsets; encoders label the first sub-token. KD losses are computed over the 9 BIO labels at the labeled sub-token per word.
- **MTA (`nerkd/mta.py`, `nerllm/mta.py`)**: layer pairs are `teacher:student` indices into `hidden_states` (0 = embeddings). Spans (every word + spaCy noun/verb phrases) are precomputed once per sentence on the gold words and cached in `cache/`. Layer pairs must match the model depths — they differ per model pair (BERT→DistilBERT `6:3 | 10:5 12:6`, Qwen `14:12 | 23:20 28:24`, MedGemma `31:17 | 52:28 62:34`). Projectors train with lr 5e-4.
- **nerllm specifics**: teacher sees few-shot prefix + target tokens, student sees zero-shot prompt + the same target tokens, so supervised positions line up and logits are compared token by token (vocab cut to the smaller output layer if sizes differ); the loss is only on JSON tokens + EOS. Teacher/student have separate token→span maps for MTA because of the prefix. Student is LoRA-trained then merged.
- Single-seed results differ by <0.3 F1 between methods; use multi-seed (IWD) runs before drawing conclusions.
