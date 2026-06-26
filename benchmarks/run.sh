#!/usr/bin/env bash
# One-command UFM benchmark: run the sweep, then generate figures.
set -euo pipefail
cd "$(dirname "$0")"
python run_benchmark.py "$@"
python plot_results.py
echo "Done. See results/ for summary.json + figures."
