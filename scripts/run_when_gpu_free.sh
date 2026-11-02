#! /bin/bash
# Poll until the GPU has enough free memory (and the gated MedGemma models are accessible, and the data is prepared),
# then run run_medgemma.sh once. Log -> logs/llm/watcher.log
#   nohup bash scripts/run_when_gpu_free.sh > /dev/null 2>&1 &
# Environment: NEED_GB (free GPU memory required, default 95), MAX_UTIL (max GPU util %, default 10),
#   STABLE (consecutive OK polls required, default 3), INTERVAL (seconds between polls, default 120),
#   EXPS / SMOKE / EPOCHS ... are passed through to run_medgemma.sh.
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
NEED_GB=${NEED_GB:-95}; MAX_UTIL=${MAX_UTIL:-10}; STABLE=${STABLE:-3}; INTERVAL=${INTERVAL:-120}
VENV=${VENV:-${ROOT}/.venv-gemma}
TEACHER=${TEACHER:-google/medgemma-27b-it}; STUDENT=${STUDENT:-google/medgemma-4b-it}
mkdir -p logs/llm
LOG=logs/llm/watcher.log
exec 9>logs/llm/watcher.lock; flock -n 9 || { echo "watcher already running" >&2; exit 1; }
say() { echo "[$(date '+%F %T')] $*" >> "${LOG}"; }

gpu_ok() {  # free memory >= NEED_GB and util <= MAX_UTIL on GPU ${GPU:-0}
  local line free util
  line=$(nvidia-smi -i "${GPU:-0}" --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || return 1
  IFS=', ' read -r total used util <<< "${line}"
  free=$(( (total - used) / 1024 ))
  say "gpu free=${free}GB util=${util}%"
  [ "${free}" -ge "${NEED_GB}" ] && [ "${util}" -le "${MAX_UTIL}" ]
}
hf_ok() {
  "${VENV}/bin/python" - "${TEACHER}" "${STUDENT}" <<'PY' >> "${LOG}" 2>&1
import sys
from huggingface_hub import hf_hub_download
for m in sys.argv[1:]:
    hf_hub_download(m, "config.json")
PY
}
data_ok() {
  [ -f cache/envimed/train.jsonl ] && [ -f cache/envimed/dev.jsonl ] && [ -f cache/envimed/test.jsonl ] && return 0
  say "preparing data"
  PATH="${VENV}/bin:${PATH}" bash scripts/run_llm.sh prepare >> "${LOG}" 2>&1
}

ok=0
say "watcher started: need ${NEED_GB}GB free, util<=${MAX_UTIL}%, ${STABLE} consecutive polls every ${INTERVAL}s"
while true; do
  if gpu_ok; then ok=$((ok + 1)); else ok=0; fi
  if [ "${ok}" -ge "${STABLE}" ]; then
    if ! hf_ok; then say "no access to gated models yet (accept licences on huggingface.co/google/medgemma-27b-it and -4b-it)"; ok=0
    elif ! data_ok; then say "data preparation failed (rate limit?), will retry"; ok=0
    else
      say "GPU free: starting run_medgemma.sh"
      VENV="${VENV}" bash run_medgemma.sh >> logs/llm/run_medgemma.log 2>&1
      say "run_medgemma.sh exited with $?"
      exit 0
    fi
  fi
  sleep "${INTERVAL}"
done
