#!/bin/sh
# Regenerate every figure and the graphical abstract, then run the numerical checks.
set -e
cd "$(dirname "$0")/figures"
for f in fig_setup fig_decay fig_fringe fig_casimir graphical_abstract; do
    python3 "$f.py"
done
cd ../numerics
python3 verify.py
