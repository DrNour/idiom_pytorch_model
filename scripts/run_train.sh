#!/usr/bin/env bash
set -e

python -m src.train \
  --data_path data/sample_idiom_translation_errors.csv \
  --model_name xlm-roberta-base \
  --output_dir outputs/idiom_error_model \
  --epochs 3 \
  --batch_size 4 \
  --max_length 192
