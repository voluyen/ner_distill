#! /bin/bash
# IWD protocol (arXiv 2609.22566): train on CoNLL-2003, report CoNLL test + OOD test on WNUT-17 and
# OntoNotes 5.0, mean ± std over seeds. Same-tokenizer pairs:
#   qwen    : Qwen/Qwen2.5-7B -> Qwen/Qwen2.5-0.5B (default). Teacher = LoRA SFT, merged; student = full FT.
#             Causal decoders: labels on the last sub-token of each word. MTA 14:12 | 23:20 28:24.
#   bert    : bert-base-cased            -> distilbert-base-cased   (MTA 6:3 | 10:5 12:6)
#   deberta : microsoft/deberta-v3-base  -> microsoft/deberta-v3-xsmall (MTA 6:6 | 10:10 12:12, the IWD pair)
# One teacher per pair (seed 42); every student config runs once per seed.
#   bash scripts/run_iwd.sh                       # everything
#   PAIRS=bert SEEDS="1 2" JOBS=4 bash scripts/run_iwd.sh student amid
# Output: results_iwd/<pair>/<group>/<name>_s<seed>/, logs: logs/iwd/. Finished runs are skipped.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PAIRS=${PAIRS:-qwen}
SEEDS=${SEEDS:-1 2 3 4 5}
JOBS=${JOBS:-2}          # concurrent runs on the GPU
EXPS=("$@")
[ ${#EXPS[@]} -eq 0 ] && EXPS=(student fkl skl srkl amid srkl_mta amid_mta)
mkdir -p logs/iwd

pair_env() {  # pair -> environment for scripts/run.sh
  case "$1" in
    qwen)    echo "RESULTS=results_iwd/qwen TEACHER=Qwen/Qwen2.5-7B STUDENT=Qwen/Qwen2.5-0.5B MTA_WORD=14:12 MTA_PHRASE='23:20 28:24' TEACHER_ARGS='${QWEN_TEACHER_ARGS:---label-pos last --lora-r 16 --grad-ckpt --lr 1e-4}' STUDENT_ARGS='${QWEN_STUDENT_ARGS:---label-pos last --teacher-bf16 --lr 1e-5}'" ;;
    bert)    echo "RESULTS=results_iwd/bert TEACHER=bert-base-cased STUDENT=distilbert-base-cased MTA_WORD=6:3 MTA_PHRASE='10:5 12:6'" ;;
    deberta) echo "RESULTS=results_iwd/deberta TEACHER=microsoft/deberta-v3-base STUDENT=microsoft/deberta-v3-xsmall MTA_WORD=6:6 MTA_PHRASE='10:10 12:12'" ;;
    *) echo "unknown pair $1" >&2; exit 1 ;;
  esac
}

run_one() {  # pair exp seed
  local pair=$1 exp=$2 seed=$3 env out suffix
  env=$(pair_env "${pair}")
  suffix=$([ "${exp}" = teacher ] && echo "" || echo "_s${seed}")
  out=$(eval "${env} bash scripts/run.sh list" | awk -v e="${exp}" '$1==e {print $3}')${suffix}
  if [ -f "${out}/results.json" ]; then echo "[skip] ${pair} ${exp} ${seed}"; return 0; fi
  local log="logs/iwd/${pair}_${exp}${suffix}.log"
  echo "[$(date '+%F %T')] start ${pair} ${exp} seed ${seed}"
  if eval "${env} SEED=${seed} SUFFIX=${suffix} bash scripts/run.sh ${exp}" > "${log}" 2>&1; then
    echo "[$(date '+%F %T')] done  ${pair} ${exp} seed ${seed}: $(grep -E '^(BEST|OOD)' "${log}" | tr '\n' ' ')"
  else
    echo "[$(date '+%F %T')] FAIL  ${pair} ${exp} seed ${seed}: see ${log}"; return 1
  fi
}
export -f run_one pair_env

# 1. teachers (seed 42), in parallel across pairs
for p in ${PAIRS}; do echo "${p} teacher 42"; done | xargs -P "${JOBS}" -L 1 bash -c 'run_one "$@"' _ || exit 1
# 2. students
for s in ${SEEDS}; do for p in ${PAIRS}; do for e in "${EXPS[@]}"; do echo "${p} ${e} ${s}"; done; done; done \
  | xargs -P "${JOBS}" -L 1 bash -c 'run_one "$@"' _
status=$?
.venv/bin/python -m nerkd.results_iwd results_iwd | tee results_iwd/RESULTS.md
exit ${status}
