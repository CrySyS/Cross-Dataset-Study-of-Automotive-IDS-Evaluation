#!/bin/bash
# Example: Custom window sizes

.venv/bin/python -m window_size_hyperparameter_module \
    --dataset_name "MyDataset" \
    --train_glob "data_parquet/path/to/train/*.parquet" \
    --test_glob "data_parquet/path/to/test/*.parquet" \
    --window_sizes "5,10,15,20,30,60,120,300" \
    --output_dir "window_size_hyperparameter_module/results"

