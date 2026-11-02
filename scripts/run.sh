#! /bin/bash
# Run one named experiment. Output -> results/<group>/<name>[SUFFIX].
#   bash scripts/run.sh <experiment> [extra args for nerkd.train ...]
#   bash scripts/run.sh list                      # show all experiments
# Environment overrides: TEACHER, STUDENT, TEACHER_CKPT, SEED, GPU, SUFFIX,
#   RESULTS (output root, default results), MTA_WORD / MTA_PHRASE (MTA layer pairs), OOD (datasets for OOD eval),
#   TEACHER_ARGS / STUDENT_ARGS (extra nerkd.train args for the teacher run / every other run)
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TEACHER=${TEACHER:-bert-base-cased}
STUDENT=${STUDENT:-distilbert-base-cased}
RESULTS=${RESULTS:-results}
TEACHER_CKPT=${TEACHER_CKPT:-${RESULTS}/sft/${TEACHER//\//_}}
SEED=${SEED:-42}
MTA_WORD=${MTA_WORD:-6:3}
MTA_PHRASE=${MTA_PHRASE:-10:5 12:6}
OOD=${OOD-wnut17 ontonotes}

COMMON="--lr 5e-5 --epochs 5 --batch-size 32 --bf16 --seed ${SEED} ${OOD:+--ood-eval ${OOD}}"
KD="--model ${STUDENT} --teacher ${TEACHER_CKPT} --kd-ratio 0.5"
AMID="--kd-type amid --amid-div ab --amid-order pr --amid-alpha 0.5 --amid-lam 0.5"
MTA="--w-mta 2.0 --mta-word-layers '${MTA_WORD}' --mta-phrase-layers '${MTA_PHRASE}'"

# name | output dir | arguments
EXPERIMENTS=(
  "teacher        | sft/${TEACHER//\//_}     | --model ${TEACHER}"
  "student        | sft/${STUDENT//\//_}     | --model ${STUDENT}"
  "fkl            | kd/fkl                   | ${KD} --kd-type fkl"
  "rkl            | kd/rkl                   | ${KD} --kd-type rkl"
  "skl            | kd/skl                   | ${KD} --kd-type skl --skew-alpha 0.1"
  "srkl           | kd/srkl                  | ${KD} --kd-type srkl --skew-alpha 0.1"
  "amid           | kd/amid                  | ${KD} ${AMID}"
  "amid_mta       | kd/amid_mta              | ${KD} ${AMID} ${MTA}"
  "srkl_mta       | kd/srkl_mta              | ${KD} --kd-type srkl --skew-alpha 0.1 ${MTA}"
  "mta_only       | kd/mta_only              | ${KD} ${MTA}"
  "abl_word_only  | ablation/word_only       | ${KD} ${AMID} --w-mta 2.0 --mta-word-layers '6:3 10:5 12:6' --mta-phrase-layers ''"
  "abl_phrase_only| ablation/phrase_only     | ${KD} ${AMID} --w-mta 2.0 --mta-word-layers '' --mta-phrase-layers '6:3 10:5 12:6'"
  "abl_no_weight  | ablation/no_weight       | ${KD} ${AMID} ${MTA} --mta-no-weight"
)

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }

EXP=${1:?usage: bash scripts/run.sh <experiment|list> [extra args]}; shift
if [ "${EXP}" = "list" ]; then
  for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n o a <<< "$e"; printf "%-16s -> ${RESULTS}/%s\n" "$(trim "$n")" "$(trim "$o")"; done
  exit 0
fi
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r n o a <<< "$e"
  [ "$(trim "$n")" = "${EXP}" ] || continue
  OUT="${RESULTS}/$(trim "$o")${SUFFIX}"
  [ -n "${GPU}" ] && export CUDA_VISIBLE_DEVICES=${GPU}
  [ -x .venv/bin/python ] && export PATH="${ROOT}/.venv/bin:${PATH}"
  export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=True
  if [[ "${EXP}" != teacher && "${EXP}" != student && ! -f "${TEACHER_CKPT}/config.json" ]]; then
    echo "teacher checkpoint ${TEACHER_CKPT} not found: run 'bash scripts/run.sh teacher' first" >&2; exit 1
  fi
  EXTRA=$([ "${EXP}" = teacher ] && echo "${TEACHER_ARGS}" || echo "${STUDENT_ARGS}")
  CMD="python -m nerkd.train $(trim "$a") ${COMMON} ${EXTRA} --save ${OUT}"
  echo "[$(date '+%F %T')] ${EXP}: ${CMD} $*"
  eval "${CMD}" '"$@"'
  exit 0
done
echo "unknown experiment '${EXP}'. Available:" >&2; bash "$0" list >&2; exit 1
