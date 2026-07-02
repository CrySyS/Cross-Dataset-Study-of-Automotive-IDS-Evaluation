# CANShield

This is the modified version of the original CANShield (https://github.com/shahriar0651/CANShield/), to support new datasets. Unchanged files were removed.



## 4. Full ROAD Training (Final Run)

Run this when you want final model artifacts for ROAD.

```bash
python run_development_canshield.py \
  --config-name road \
  per_of_samples=1.0 \
  debug_input_pipeline=false \
  debug_outputs=false
```

## 5. Full ROAD Evaluation (Thresholds + Test Predictions)

Run evaluation after training.

```bash
python run_evaluation_canshield.py \
  --config-name road \
  per_of_samples=1.0
```

## 6. Optional: Save Logs While Running

Training log:

```bash
python run_development_canshield.py --config-name road per_of_samples=1.0 \
  debug_input_pipeline=false debug_outputs=false \
  2>&1 | tee /tmp/canshield_road_train_$(date +%Y%m%d_%H%M%S).log
```

Evaluation log:

```bash
python run_evaluation_canshield.py --config-name road per_of_samples=1.0 \
  2>&1 | tee /tmp/canshield_road_eval_$(date +%Y%m%d_%H%M%S).log
```