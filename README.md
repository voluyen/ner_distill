# ner_distill

Knowledge distillation from an encoder to another encoder for NER (token classification) on **CoNLL-2003**.
One training script covers several families of distillation losses:

| Family | `--kd-type` / flag | Source |
|---|---|---|
| Hinton KD / KL variants | `fkl`, `rkl`, `sym`, `js`, `tv` | Hinton et al., 2015 |
| DistiLLM | `skl`, `srkl` (skew KL, α = 0.1) | Ko et al., ICML 2024 · [jongwooko/distillm](https://github.com/jongwooko/distillm) |
| AMiD | `amid` (α-mixture assistant distribution) | Shin et al., ICLR 2026 · [voluyen/amid_mta](https://github.com/voluyen/amid_mta) |
| MTA | `--w-mta > 0`, combinable with any `--kd-type` | word/phrase span alignment · [voluyen/amid_mta](https://github.com/voluyen/amid_mta) |

Default setup: teacher `bert-base-cased` (108M) → student `distilbert-base-cased` (65M). The two share one
WordPiece vocab, so each batch is tokenized once and fed to both models. Metric: entity-level micro F1 (`seqeval`).
The checkpoint with the best dev F1 is evaluated on test.

## Layout
```
nerkd/
  data.py      CoNLL-2003 loading, sub-token label alignment, spaCy phrase spans (cached in cache/)
  losses.py    logit divergences: DistiLLM (fkl/rkl/skl/srkl/...) + AMiD
  mta.py       MTA span alignment of hidden states (word + phrase level)
  train.py     single training entry point (SFT, KD, AMiD, MTA)
  ood.py       OOD evaluation on WNUT-17 / OntoNotes 5.0 with CoNLL label mapping (IWD protocol)
  results.py   markdown table from every results/**/results.json
  results_iwd.py  mean ± std over seeds (CoNLL test + OOD) for results_iwd/
scripts/
  run.sh       run one named experiment (registry of all configs in one place)
  run_all.sh   run a list of experiments, skip finished ones, print the table
  run_iwd.sh   IWD protocol: 2 model pairs x configs x 5 seeds, CoNLL + OOD eval, run in parallel
nerllm/       generative NER (text -> JSON) on En-ViMedNER, MedGemma-27B (no SFT) -> MedGemma-4B KD (see nerllm/README.md)
scripts/run_llm.sh   prepare data + run its experiments
setup.sh       create .venv with uv (torch, transformers, seqeval, spaCy + en_core_web_sm)
```
Outputs: `results/<group>/<name>/` (checkpoint, `log.txt`, `results.json`), `logs/<name>.log`. Everything in
`results/`, `logs/`, `cache/` and `third_party/` is git-ignored.

## Quick start
```bash
bash setup.sh                         # one-time: create .venv
bash scripts/run.sh list              # list available experiments
bash scripts/run.sh teacher           # 1. fine-tune the teacher (required for all KD runs)
bash scripts/run.sh amid_mta          # 2. run any experiment
bash scripts/run_all.sh               # or: the full table (skips runs that already have results.json)
bash scripts/run_all.sh amid amid_mta # or: a subset
.venv/bin/python -m nerkd.results results
```
Extra arguments go straight to `nerkd.train`, e.g. a quick debug run:
`SUFFIX=_debug bash scripts/run.sh amid_mta --train-num 800 --eval-num 300 --epochs 1`.
Environment overrides: `GPU`, `SEED`, `SUFFIX` (appended to the output dir), `TEACHER`, `STUDENT`, `TEACHER_CKPT`,
`RESULTS` (output root), `MTA_WORD` / `MTA_PHRASE` (MTA layer pairs), `OOD` (OOD datasets, default
`wnut17 ontonotes`; set `OOD=` to skip).

## Experiments (`scripts/run.sh`)
Shared settings: lr 5e-5, 5 epochs, batch 32, bf16, seed 42, `kd_ratio` 0.5, T = 1.

| name | loss |
|---|---|
| `teacher`, `student` | CE on gold labels only |
| `fkl` `rkl` `skl` `srkl` | 0.5·CE + 0.5·KD |
| `amid` | 0.5·CE + 0.5·AMiD (div `ab` (α_ab = 0.2, β_ab = 0.7), order `pr`, α = 0.5, λ = 0.5) |
| `amid_mta` | 0.5·CE + 0.5·(AMiD + 2.0·MTA) |
| `srkl_mta`, `mta_only` | MTA on top of SRKL, and MTA without a logit KD term |
| `abl_word_only` / `abl_phrase_only` | MTA using only word spans / only phrase spans |
| `abl_no_weight` | MTA with uniform token weights instead of attention-based importance |

To add a config, add one line to `EXPERIMENTS` in `scripts/run.sh`.

## IWD protocol (`scripts/run_iwd.sh`)
Follows IWD (Kannangara & Dutta, arXiv 2609.22566): train on CoNLL-2003 only, report CoNLL test F1 plus
out-of-distribution F1 on the **test** splits of WNUT-17 (`tner/wnut2017`) and OntoNotes 5.0 (`tner/ontonotes5`),
as mean ± std over seeds 1–5. One teacher per pair (seed 42). Pairs (`PAIRS=...`):
- `qwen` (default): **Qwen2.5-7B → Qwen2.5-0.5B**. The 7B teacher is LoRA-tuned (`--lora-r 16`, all linear layers,
  head trained in full) on a bf16 base, then merged; the 0.5B student is fully fine-tuned and the frozen teacher
  runs in bf16 (`--teacher-bf16`). Both are causal, so each word's label sits on its **last** sub-token
  (`--label-pos last`). Qwen's byte-level BPE is fed the whole sentence, and tokens are mapped back to words by
  character offsets (`nerkd.data.encode_words`). MTA layers `14:12 | 23:20 28:24` (28 → 24 layers).
- `bert` (bert-base-cased → distilbert-base-cased) and `deberta` (deberta-v3-base → deberta-v3-xsmall, the encoder
  pair used by IWD; MTA layers `6:6 | 10:10 12:12`).
```bash
bash scripts/run_iwd.sh                                   # Qwen pair, all configs x 5 seeds -> results_iwd/RESULTS.md
PAIRS=bert SEEDS="1 2" JOBS=4 bash scripts/run_iwd.sh amid # subset
python -m nerkd.ood <ckpt_dir> ...                        # OOD eval of an existing checkpoint
```
IWD does not publish its label mapping. Ours (`nerkd/ood.py`): WNUT person→PER, location→LOC,
corporation/group→ORG, product/creative-work ignored; OntoNotes PERSON→PER, ORG→ORG, GPE/LOC→LOC,
NORP/FAC/EVENT/WORK_OF_ART/LAW/LANGUAGE/PRODUCT ignored, numeric types (DATE, MONEY, ...) are O. "Ignored" gold
spans are removed together with any predicted span overlapping them; predicted MISC is dropped. Score = exact
span + type micro F1.

Results for Qwen2.5-7B → 0.5B (5 seeds, entity F1; full write-up in `reports/`):

| run | CoNLL test | WNUT-17 | OntoNotes |
|---|---|---|---|
| teacher Qwen2.5-7B (LoRA) | 85.83 | 48.38 | 73.90 |
| student 0.5B, no KD | 83.57 ± 0.48 | 44.51 ± 2.07 | 70.99 ± 0.13 |
| fkl | 83.89 ± 0.29 | 43.66 ± 2.49 | 71.05 ± 0.34 |
| skl | 83.72 ± 0.28 | 43.68 ± 1.98 | 71.23 ± 0.90 |
| srkl | 84.14 ± 0.19 | 43.76 ± 2.30 | 71.07 ± 0.28 |
| amid | 83.44 ± 0.40 | 42.91 ± 2.48 | 70.88 ± 0.76 |
| srkl_mta | 84.13 ± 0.38 | 43.38 ± 1.75 | 71.56 ± 0.56 |
| amid_mta | 84.35 ± 0.27 | 43.20 ± 2.40 | 71.67 ± 0.55 |

## How AMiD and MTA are adapted to NER
- **AMiD**: the loss `amid()` is the same as in `amid_mta`. It is applied over the 9 BIO labels at the first
  sub-token of each word, instead of over the vocabulary at every generated token. The adaptive student generation
  and replay buffer (`adaptive-amid`) only apply to autoregressive LMs, so they are not used here.
- **MTA** (`nerkd/mta.py`): the loss is the same as `span_utils.py`: an attention-based importance weight for each
  token, a weighted mean for each span, an MSE between the span-span cosine-similarity matrices of student and
  teacher, and `+ 0.1 · (1 − cos(projector(student token), teacher token))`. Projectors use lr 5e-4.
  What changed:
  - Spans are **precomputed once per sentence** on the gold CoNLL words. spaCy runs on `Doc(words=...)`, so no
    re-tokenization or char-offset matching happens each step. The result is cached in `cache/`.
  - Word spans = **every word** (all its sub-tokens). Phrase spans = noun chunks + verb phrases
    (`AUX* ADV* VERB+ ADV*`), non-overlapping.
  - Layer pairs are `teacher:student` indices into `hidden_states` (0 = embeddings). The default maps BERT 12 layers
    to DistilBERT 6 layers: word `6:3`, phrase `10:5 12:6`. This mirrors GPT-2 `24/36/48 → 6/9/12`
    (1 word pair + 2 phrase pairs).

## Results (CoNLL-2003, seed 42)
All runs: seed 42. After `run_all.sh`, the table is also written to `results/RESULTS.md`.

| run | params (M) | dev F1 | test P | test R | test F1 |
|---|---|---|---|---|---|
| teacher (BERT-base) | 107.7 | 95.2 | 90.26 | 92.03 | 91.14 |
| student, no KD | 65.2 | 93.98 | 89.22 | 90.28 | 89.75 |
| fkl | 65.2 | 94.09 | 88.86 | 90.81 | 89.82 |
| rkl | 65.2 | 94.07 | 89.07 | 90.76 | 89.91 |
| skl (DistiLLM) | 65.2 | 94.14 | 88.84 | 90.78 | 89.8 |
| srkl (DistiLLM) | 65.2 | 93.99 | 89.54 | 90.49 | 90.01 |
| amid | 65.2 | 94.14 | 88.91 | 90.95 | 89.92 |
| amid_mta | 65.2 | 94.4 | 88.48 | 90.67 | 89.56 |
| srkl_mta | 65.2 | 94.42 | 88.76 | 90.56 | 89.65 |
| mta_only | 65.2 | 94.26 | 88.31 | 90.56 | 89.42 |
| abl_word_only | 65.2 | 94.33 | 88.5 | 90.78 | 89.63 |
| abl_phrase_only | 65.2 | 94.2 | 88.67 | 90.62 | 89.63 |
| abl_no_weight | 65.2 | 94.23 | 89.1 | 90.7 | 89.89 |

Single seed: most student runs differ by < 0.3 F1 from each other, so report averages over several seeds (`SEED=... SUFFIX=_s...`) before drawing conclusions. MTA gives the best dev F1 (94.4) but does not improve test F1 at seed 42.

## Reference code
The upstream repositories are not needed to run anything, but can be cloned into `third_party/` for reference:
```bash
git clone https://github.com/jongwooko/distillm third_party/distillm
git clone https://github.com/voluyen/amid_mta third_party/amid_mta
```

## One-command MedGemma run (DistiLLM, AMiD, MTA)
`run_medgemma.sh` (repo root) sets up a Gemma-capable env (`.venv-gemma`), checks access to the gated models, prepares
En-ViMedNER, runs a tiny smoke test, then trains MedGemma-27B -> MedGemma-4B with SKL, SRKL (DistiLLM), AMiD, MTA only,
SRKL+MTA and AMiD+MTA, and writes `results_llm/RESULTS.md`.
```bash
export HF_TOKEN=hf_xxx && bash run_medgemma.sh      # EXPS="amid amid_mta" to run a subset
```
MTA for the causal LM (`nerllm/mta.py`) aligns hidden states of the sentence tokens (spaCy words / noun+verb phrases);
teacher and student have separate token->span maps because the teacher prompt carries a few-shot prefix.
Layer pairs (teacher:student hidden-state index) default to `31:17` (word) and `52:28 62:34` (phrase).
