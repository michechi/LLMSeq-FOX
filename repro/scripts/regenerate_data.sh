#!/bin/bash
# Regenerate the three core datasets from seed.
#
# Note: the canonical CSVs ship in data/simulation/tested/. This script regenerates
# them from the original generator code; depending on minor RNG ordering, output
# may differ slightly from the canonical files. Treat regeneration as best-effort.

set -euo pipefail
REPRO_ROOT="${REPRO_ROOT:-$PWD}"
cd "$REPRO_ROOT"

mkdir -p data/simulation/tested

echo "[regenerate_data] Tricky Random (tag _9) ..."
python -m src.generators.properties_9

echo "[regenerate_data] Tricky Deterministic (tag _6) and Parity (tag test_just_pair) ..."
python -m src.generators.test_simulation_det

echo "[regenerate_data] done. CSVs in data/simulation/tested/"
ls data/simulation/tested/ | head
