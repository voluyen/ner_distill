# nerllm: generative NER on En-ViMedNER (text → JSON)

A causal LM (student: `google/medgemma-4b-it`) is trained by distillation with plain **next-token prediction** to read a sentence and write its entities as JSON.
Data: English side of [nhuvo/En-ViMedNER](https://huggingface.co/datasets/nhuvo/En-ViMedNER) (biomedical, 21 UMLS
semantic types, derived from MedMentions ST21pv): 27,013 / 8,932 / 8,947 train / dev / test sentences (+ 450 mini-test).

```
Extract all biomedical named entities from the text. Answer with a JSON list of objects with the keys "entity" (the exact span from the text) and "type".

Text: Our aim was to verify the occurrence of selected mutations of the EZH2 and ZFX genes ...
Entities:
[{"entity": "mutations", "type": "Biologic Function"}, {"entity": "EZH2", "type": "Anatomical Structure"}, ...]<end_of_turn>
```
The prompt is wrapped in the model's chat template (`<start_of_turn>user ... <start_of_turn>model`), the answer ends with `<end_of_turn>`.
Use `--no-chat` for base models.
- Input = prompt + sentence, output = JSON sorted by position, `[]` when there is no entity. The loss is only on the
  JSON tokens + EOS (the prompt is masked with -100). Types are written as names (e.g. `T103` → "Chemical")
  rather than TUI codes, see `UMLS_TYPES` in `data.py`.
- Spans are read from the sentence by character offsets, so every `entity` is an exact substring of the input.
- Sentences longer than `--max-length` tokens (default 768) are dropped from training (a truncated target would teach
  invalid JSON). They are never dropped from dev/test.

## Teacher / student
| | model | role |
|---|---|---|
| teacher | `google/medgemma-27b-it` | **no SFT**: used as is, with a few-shot prompt (`--teacher-shots`, default 5 fixed training sentences covering many types) |
| student | `google/medgemma-4b-it` | no stand-alone SFT run; trained only by distillation from the teacher (`--kd-ratio` 0.5: (1-r)·CE on gold + r·KD; `--kd-ratio 1.0` = pure KD) |

Both share the Gemma 3 tokenizer (vocab 262,144). The teacher sees `few-shot prompt + the same target tokens`, the student
`zero-shot prompt + target`; the supervised positions are identical, so teacher/student logits are compared token by token.
Gemma 3 needs `transformers>=4.50`: install a separate env from `requirements-gemma.txt` (the `.venv` of the encoder
pipeline pins 4.42.4). The checkpoints are gated on Hugging Face (accept the licence, `huggingface-cli login`).

## Run
```bash
bash scripts/run_llm.sh prepare         # one-time: download + convert -> cache/envimed/{train,dev,test}.jsonl
bash scripts/run_llm.sh list
bash scripts/run_llm.sh teacher-eval    # teacher baseline: 5-shot generation, no training
bash scripts/run_llm.sh student-eval    # student before any training (5-shot)
bash scripts/run_llm.sh srkl            # KD from the teacher (fkl|rkl|skl|srkl|amid)
```
Extra arguments go to `nerllm.train`, e.g. a smoke run:
`SUFFIX=_debug bash scripts/run_llm.sh srkl --train-num 200 --dev-num 100 --test-num 100 --epochs 1`.
Environment: `TEACHER`, `STUDENT`, `SEED`, `GPU`, `SUFFIX`, `RESULTS` (default `results_llm`), `SHOTS`, `FULL_FT=1`, `DRY=1`
(print the command only).

## Training and evaluation
- `train.py`: custom loop, bf16 autocast, cosine schedule, grad accumulation. Student = LoRA (r=64) on the
  language model of a bf16 base, merged into a plain checkpoint at the end (`FULL_FT=1`: full fine-tuning in fp32, ~70GB
  with AdamW for 4B). Add `--types-in-prompt` to list the 21 type names in the prompt.
- **Distillation** (`--teacher`): token-level KD on the JSON tokens, `(1-r)·CE + r·KD`, reusing `nerkd/losses.py`
  (`fkl rkl skl srkl sym js tv amid`). The vocab dimension is cut to the smaller output layer if the two differ. Only the divergences are used, not DistiLLM's adaptive
  student-generated outputs.
- Each epoch: dev loss on the full dev set + greedy-decoding F1 on the first `--gen-eval-num` (500) dev sentences. The
  best epoch is kept, then scored on the **full dev and test** sets. Outputs: `results.json`, `log.txt`,
  `test_predictions.jsonl` (text, gold, raw generation).
- Metric (`metrics.py`): micro P/R/F1 over (entity text, type) pairs per sentence, compared as multisets. Also
  `span_f1` (type ignored) and `valid_json` (% of generations that are one fully valid JSON list). Malformed output is
  salvaged object by object, so one broken entry does not zero a sentence.

## Checked without training
Data conversion on the mini-test split (450 sentences, 2,086 entities, 0 offset mismatches), tokenization / label masking,
chat-template prompt and few-shot teacher prompt alignment (same supervised positions as the student), metric edge cases,
all KD losses with mismatched vocab sizes, the CLI of every experiment, and the dev-loss + generation path on a tiny random
Qwen2 model. **Not verified**: anything that needs the Gemma 3 classes (not installable in the current `.venv`, and the
checkpoints are gated): loading `Gemma3ForConditionalGeneration` as a text-only LM, the `token_type_ids` workaround in
`forward()`, and the LoRA target regex. Run `srkl --train-num 50 --dev-num 20 --test-num 20 --epochs 1` first.
**No training run has been done**; hyper-parameters are untuned starting points.
Memory: teacher 27B in bf16 = ~54GB; with the 4B student (LoRA) expect ~70-90GB. The KD loss tensors are
(response tokens x 262k vocab), hence `--batch-size 4` with `--grad-accum 8`; lower the batch size if memory is tight.
