#!/bin/bash
# Example: Car-Hacking Window Size Study

.venv/bin/python -m window_size_hyperparameter_module \
    --dataset_name "Car-Hacking" \
    --train_glob "data_parquet/03_Car-HackingDataset/normal_run_data.parquet" \
    --test_glob "data_parquet/03_Car-HackingDataset/*attack.parquet" \
    --window_sizes "0.01,0.1,0.5,1,5,10,30,60,120,300" \
    --output_dir "window_size_hyperparameter_module/results_eval_on_attack_only"
