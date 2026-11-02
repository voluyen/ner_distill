#! /bin/bash
# Generative NER on En-ViMedNER (text -> JSON). Output -> ${RESULTS}/<group>/<name>[SUFFIX].
#   bash scripts/run_llm.sh prepare               # download + convert train/dev/test (one-time)
#   bash scripts/run_llm.sh <experiment> [extra args for nerllm.train ...]
#   bash scripts/run_llm.sh list
# Teacher (google/medgemma-27b-it) is NOT fine-tuned: it is used as is, with a few-shot prompt, for KD.
# Environment overrides: TEACHER, STUDENT, SEED, GPU, SUFFIX, RESULTS (default results_llm),
#   FULL_FT=1 (student: full fine-tuning instead of LoRA; ~70GB for 4B in fp32 + AdamW, plus the 27B teacher),
#   DRY=1 (print the command only),
#   SHOTS (few-shot examples in the teacher prompt, default 5)
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TEACHER=${TEACHER:-google/medgemma-27b-it}
STUDENT=${STUDENT:-google/medgemma-4b-it}
RESULTS=${RESULTS:-results_llm}
SEED=${SEED:-42}
SHOTS=${SHOTS:-5}

COMMON="--bf16 --attn eager --seed ${SEED}"
if [ "${FULL_FT}" = 1 ]; then
  TRAIN="--lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 8 --gradient-checkpointing"
else  # LoRA on the language model, merged into a plain checkpoint after training
  TRAIN="--lora-r 64 --lora-alpha 128 --lr 1e-4 --epochs 3 --batch-size 4 --grad-accum 8 --gradient-checkpointing"
fi
# KD memory: batch-size x ~100 response tokens x 262k vocab floats per loss tensor -> keep batch-size small
KD="--model ${STUDENT} --teacher ${TEACHER} --teacher-shots ${SHOTS} --kd-ratio 0.5 ${TRAIN}"

# MTA: hidden-state indices teacher:student (MedGemma-27B has 62 layers, 4B has 34); word level / phrase level
MTA_WORD=${MTA_WORD:-31:17}
MTA_PHRASE=${MTA_PHRASE:-52:28 62:34}
MTA="--w-mta 2.0 --mta-word-layers '${MTA_WORD}' --mta-phrase-layers '${MTA_PHRASE}'"

# name | output dir | arguments
EXPERIMENTS=(
  "teacher-eval | zeroshot/${TEACHER//\//_} | --eval-only --model ${TEACHER} --eval-shots ${SHOTS} --eval-batch-size 16"
  "student-eval | zeroshot/${STUDENT//\//_} | --eval-only --model ${STUDENT} --eval-shots ${SHOTS}"
  "fkl     | kd/fkl               | ${KD} --kd-type fkl"
  "rkl     | kd/rkl               | ${KD} --kd-type rkl"
  "skl     | kd/skl               | ${KD} --kd-type skl --skew-alpha 0.1"
  "srkl    | kd/srkl              | ${KD} --kd-type srkl --skew-alpha 0.1"
  "amid    | kd/amid              | ${KD} --kd-type amid"
  "mta_only | kd/mta_only         | ${KD} --kd-type none ${MTA}"
  "srkl_mta | kd/srkl_mta         | ${KD} --kd-type srkl --skew-alpha 0.1 ${MTA}"
  "amid_mta | kd/amid_mta         | ${KD} --kd-type amid ${MTA}"
)

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }

EXP=${1:?usage: bash scripts/run_llm.sh <prepare|list|experiment> [extra args]}; shift
[ -n "${VENV}" ] && export PATH="${VENV}/bin:${PATH}"  # e.g. VENV=.venv-gemma (see run_medgemma.sh)
[ -x .venv/bin/python ] && export PATH="${ROOT}/.venv/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=True
if [ "${EXP}" = prepare ]; then python -m nerllm.data "$@"; exit 0; fi
if [ "${EXP}" = list ]; then
  for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n o a <<< "$e"; printf "%-10s -> ${RESULTS}/%s\n" "$(trim "$n")" "$(trim "$o")"; done
  exit 0
fi
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r n o a <<< "$e"
  [ "$(trim "$n")" = "${EXP}" ] || continue
  [ -n "${GPU}" ] && export CUDA_VISIBLE_DEVICES=${GPU}
  [ -n "${DRY}" ] || [ -f cache/envimed/train.jsonl ] || { echo "data not prepared: run 'bash scripts/run_llm.sh prepare' first" >&2; exit 1; }
  CMD="python -m nerllm.train $(trim "$a") ${COMMON} --save ${RESULTS}/$(trim "$o")${SUFFIX}"
  echo "[$(date '+%F %T')] ${EXP}: ${CMD} $*"
  [ -n "${DRY}" ] && exit 0  # DRY=1: only print the command
  eval "${CMD}" '"$@"'
  exit 0
done
echo "unknown experiment '${EXP}'. Available:" >&2; bash "$0" list >&2; exit 1
