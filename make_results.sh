#!/usr/bin/env bash
set -euo pipefail

python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline --method all
python -m src.validate_annotations
python -m src.evaluate
python -m src.export_database
python demo/run_demo.py
python demo/run_demo.py --from-db
