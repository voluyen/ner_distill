#! /bin/bash
# Resumable En-ViMedNER download that backs off on HF 429 (45k small files, limit 5000 resolvers / 5 min).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
until .venv-gemma/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nhuvo/En-ViMedNER", repo_type="dataset", local_dir="cache/envimed/raw",
                  allow_patterns=[f"{s}/*" for s in ("train", "dev", "test", "mini-test")], max_workers=4)
PY
do echo "[$(date '+%F %T')] 429/err, sleeping 300s"; sleep 300; done
echo "[$(date '+%F %T')] download complete"
PATH="$PWD/.venv-gemma/bin:$PATH" bash scripts/run_llm.sh prepare
