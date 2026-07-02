#!/bin/bash
# Example: Custom window sizes

.venv/bin/python -m window_size_hyperparameter_module \
    --dataset_name "DAGA" \
    --train_glob "data_parquet/05_DAGA_STABILI2022/DenialOfService/clean/*.parquet" \
    --test_glob "data_parquet/05_DAGA_STABILI2022/DenialOfService/infected/*.parquet" \
    --window_sizes "0.2, 0.5, 1" \
    --output_dir "window_size_hyperparameter_module/results"

