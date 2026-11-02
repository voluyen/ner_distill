#!/usr/bin/env bash
# One-shot pipeline: MedGemma-27B (teacher, not fine-tuned) -> MedGemma-4B (student) on En-ViMedNER.
# Runs DistiLLM (SKL, SRKL), AMiD, MTA (alone and combined with SRKL / AMiD), then writes the results table.
#
#   export HF_TOKEN=hf_xxx            # token of an account that accepted the Gemma / MedGemma licences
#   bash run_medgemma.sh              # (use `nohup bash run_medgemma.sh > run.log 2>&1 &` on a remote box)
#
# Optional environment:
#   EXPS="skl srkl amid mta_only srkl_mta amid_mta"   experiments to run (default below), teacher-eval/student-eval are
#                                                     the untrained baselines
#   GPU=0              CUDA_VISIBLE_DEVICES          SEED=42        FORCE=1   re-run finished experiments
#   SMOKE=0            skip the 1-minute smoke test that precedes the real runs
#   EPOCHS=3           training epochs               FULL_FT/SHOTS/...  see scripts/run_llm.sh
# Needs: Linux, one GPU with >= ~90 GB (H100/H200/A100-80GB+), ~150 GB free disk, internet.
# Output: results_llm/<group>/<name>/{results.json,log.txt,...}, results_llm/RESULTS.md, logs/llm/*.log
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$PWD
VENV=${VENV:-${ROOT}/.venv-gemma}
EXPS=${EXPS:-"teacher-eval student-eval skl srkl amid mta_only srkl_mta amid_mta"}
TEACHER=${TEACHER:-google/medgemma-27b-it}
STUDENT=${STUDENT:-google/medgemma-4b-it}
export VENV TEACHER STUDENT HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH="${ROOT}:${PYTHONPATH}"
[ -n "${GPU}" ] && export CUDA_VISIBLE_DEVICES=${GPU}
say() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment
if [ ! -x "${VENV}/bin/python" ] || ! "${VENV}/bin/python" -c "import transformers, peft, spacy, datasets; spacy.load('en_core_web_sm')" 2>/dev/null; then
  say "setting up ${VENV} (transformers>=4.50 for Gemma 3)"
  if ! command -v uv >/dev/null; then
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || die "cannot install uv"
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  uv venv "${VENV}" --python 3.11 --allow-existing || die "uv venv failed"
  uv pip install --python "${VENV}/bin/python" -r nerllm/requirements-gemma.txt || die "pip install failed"
fi
export PATH="${VENV}/bin:${PATH}"
python - <<'PY' || die "no CUDA GPU visible"
import torch, transformers
assert torch.cuda.is_available()
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPU", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
PY

# ---------------------------------------------------------------- 2. Hugging Face access (models are gated)
python - "${TEACHER}" "${STUDENT}" <<'PY' || die "no access to the gated models. Accept the licence on huggingface.co/google/medgemma-27b-it and medgemma-4b-it, then: export HF_TOKEN=hf_xxx"
import sys
from huggingface_hub import hf_hub_download
for m in sys.argv[1:]:
    hf_hub_download(m, "config.json")
    print("access ok:", m)
PY

# ---------------------------------------------------------------- 3. data (En-ViMedNER, English side)
if [ ! -f cache/envimed/train.jsonl ] || [ ! -f cache/envimed/dev.jsonl ] || [ ! -f cache/envimed/test.jsonl ]; then
  say "preparing data"
  bash scripts/run_llm.sh prepare || die "data preparation failed"
fi

# ---------------------------------------------------------------- 4. smoke test (all code paths, tiny sizes)
if [ "${SMOKE:-1}" = 1 ]; then
  say "smoke test (amid + MTA on 48 sentences)"
  mkdir -p logs/llm
  RESULTS=results_llm_smoke bash scripts/run_llm.sh amid_mta --train-num 48 --dev-num 16 --test-num 16 \
      --gen-eval-num 16 --epochs 1 --batch-size 2 --grad-accum 2 > logs/llm/smoke.log 2>&1 \
    || { tail -30 logs/llm/smoke.log; die "smoke test failed, see logs/llm/smoke.log"; }
  rm -rf results_llm_smoke
  say "smoke test passed"
fi

# ---------------------------------------------------------------- 5. experiments
mkdir -p logs/llm
RES=${RESULTS:-results_llm}
EXTRA=()
[ -n "${EPOCHS}" ] && EXTRA+=(--epochs "${EPOCHS}")
FAILED=()
for e in ${EXPS}; do
  out=$(bash scripts/run_llm.sh list | awk -v e="$e" '$1==e {print $3}')
  [ -n "${out}" ] || { echo "unknown experiment ${e}"; FAILED+=("${e}"); continue; }
  if [ -z "${FORCE}" ] && [ -f "${out}${SUFFIX}/results.json" ]; then say "[skip] ${e}: results exist"; continue; fi
  say "start ${e} (log: logs/llm/${e}.log)"
  if bash scripts/run_llm.sh "${e}" "${EXTRA[@]}" > "logs/llm/${e}.log" 2>&1; then
    say "done  ${e}: $(grep '^BEST' "logs/llm/${e}.log" | tail -1)"
  else
    say "FAIL  ${e}: see logs/llm/${e}.log"; FAILED+=("${e}")
  fi
done

# ---------------------------------------------------------------- 6. report
python -m nerllm.results "${RES}"
say "report: ${RES}/RESULTS.md"
[ ${#FAILED[@]} -eq 0 ] || { echo "failed: ${FAILED[*]}"; exit 1; }
