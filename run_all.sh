#!/bin/sh
# Regenerate every figure of the article, then run the numerical checks.
set -e
cd "$(dirname "$0")"
for f in fig_setup fig_decay fig_fringe fig_casimir; do
    python3 "$f.py"
done
python3 verify.py
