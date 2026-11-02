#! /bin/bash
# Run a list of experiments sequentially, then print the results table.
#   bash scripts/run_all.sh                       # default: full pipeline
#   bash scripts/run_all.sh amid amid_mta         # only these
# Skips an experiment whose results.json already exists (set FORCE=1 to rerun).
# Logs -> logs/<experiment>.log
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
EXPS=("$@")
[ ${#EXPS[@]} -eq 0 ] && EXPS=(teacher student fkl rkl skl srkl amid amid_mta srkl_mta mta_only
                              abl_word_only abl_phrase_only abl_no_weight)
mkdir -p logs
FAILED=()
for e in "${EXPS[@]}"; do
  out=$(bash scripts/run.sh list | awk -v e="$e" '$1==e {print $3}')
  if [ -z "${FORCE}" ] && [ -f "${out}${SUFFIX}/results.json" ]; then
    echo "[skip] ${e}: ${out}${SUFFIX}/results.json exists"; continue
  fi
  echo "[$(date '+%F %T')] start ${e} (log: logs/${e}${SUFFIX}.log)"
  if bash scripts/run.sh "${e}" > "logs/${e}${SUFFIX}.log" 2>&1; then
    echo "[$(date '+%F %T')] done  ${e}: $(grep '^BEST' "logs/${e}${SUFFIX}.log" | tail -1)"
  else
    echo "[$(date '+%F %T')] FAIL  ${e}: see logs/${e}${SUFFIX}.log"; FAILED+=("${e}")
    [ "${e}" = teacher ] && exit 1
  fi
done
.venv/bin/python -m nerkd.results "${RESULTS:-results}" | tee "${RESULTS:-results}/RESULTS.md"
[ ${#FAILED[@]} -eq 0 ] || { echo "failed: ${FAILED[*]}"; exit 1; }
