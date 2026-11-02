#! /bin/bash
# Create .venv with uv (installs uv if missing). Includes the spaCy en_core_web_sm model used by MTA.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null; then
  [ -x "${HOME}/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv sync --python 3.12
.venv/bin/python -c "import torch, spacy; spacy.load('en_core_web_sm'); print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
