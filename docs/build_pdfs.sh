#!/usr/bin/env bash
# Render the hackathon writeup (README.md) to docs/writeup.pdf: mistune -> HTML -> headless Chrome.
# Usage: docs/build_pdfs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$ROOT/.venv/bin/python" "$ROOT/docs/build_writeup.py" "$TMP/writeup.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$ROOT/docs/writeup.pdf" "file://$TMP/writeup.html" 2>/dev/null
echo "wrote docs/writeup.pdf"
