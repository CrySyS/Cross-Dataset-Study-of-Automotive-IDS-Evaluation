# Cross-Dataset-Study-of-Automotive-IDS-Evaluation
Source code for our paper: CAN We Trust Your Results? A Cross-Dataset Study of Automotive IDS Evaluation

Find the paper here: https://static.crysys.hu/v1/publications/files/KoltaiGA2026ACSW

Please cite it like this:
```
@inproceedings {
    author = {B. Koltai and G. Ács and A. Gazdag},
    title = {CAN We Trust Your Results? A Cross-Dataset Study of Automotive IDS Evaluation},
    publisher = {Euro S&P - ACSW26 workshop},
    year = {2026}
}
```

The evaluation framework implements different CAN IDS methods, runs and evaluates them in the same way on multiple datasets. You can extend methods and datasets in the registry.

## Script organization

- **Primary entrypoints**
  - `cli.py` - for evaluation
  - `convert_all.py` - for dataset conversion to same format


## Quick start (CLI)

```bash
# 0) (optional) activate virtualenv
source .venv/bin/activate

# 1) Convert raw datasets to canonical parquet (first time / when data changes)
python3 convert_all.py

# 2) Run one method on explicit train/test globs
# see registry.py for the available methods
python3 cli.py assoc_rules \
  --train_glob "data_parquet/04_CAN-IntrusionDataset_OTIDS/Attack_free_dataset.parquet" \
  --test_glob "data_parquet/04_CAN-IntrusionDataset_OTIDS/*_attack_dataset.parquet" \
  --out_dir results_single

# 3) Run full evaluation matrix (EVAL_PLAN) + auto reporting
python3 cli.py all --n_jobs 4 --out_dir results_full

# 4) Re-run reporting only on an existing results folder
python3 cli.py report --out_dir results_full
```

Notes:
- Use `--skip_report` with `all` if you only want raw evaluation outputs.
- `report` mode does not retrain models; it only regenerates tables/plots from `metrics.json` files.



## High-level

- **CLI** (`cli.py`)
  - Only: parse args, decide mode (single vs run_all), call into `eval.core` / `eval.parallel`.

- **Eval core** (`unified_ids/eval/core.py`)
  - Single (method, dataset) run.
  - Data loading, metrics, ROC plots, writing `metrics.json`.
  - Uses **BaseFlowIDS** interface (`fit`, `score_windows`, `score_messages`, optional `paper_eval`).
  - Uses **METHOD_REGISTRY** to build models.

- **Parallel orchestration** (`unified_ids/eval/parallel.py`)
  - Parallelization unit = **dataset**.
  - One process per dataset config in `EVAL_PLAN`.
  - Inside process: multiple methods share same `df_tr/df_te`.
  - Handles failure logging.

- **Eval config** (`unified_ids/eval/config.py`)
  - `EVAL_PLAN`: list of `{name, data_glob/train_glob/test_glob, methods}`.
  - `WINDOW_METHODS`, `MESSAGE_METHODS`: hints for eval semantics.

- **Method registry** (`unified_ids/methods/registry.py`)
  - `METHOD_REGISTRY[method_name] -> MethodFactory`.
  - Encapsulates constructor + default hyperparams.
  - Adding new method: implement class, register here.

- **Base class** (`BaseFlowIDS`)
  - Abstract interface for flow-based IDS.### Reporting only (`method == report`)

1. `cli.py` → `run_post_reporting(out_dir)`.
2. Rebuild tables/plots from existing `metrics.json` files under `out_dir`.
3. No model training or scoring is executed.

---

  - Methods:
    - `fit(df_train)`
    - `score_messages(df_test) -> (id, score, label)`
    - `score_windows(df_test) -> (id, score, label)`
    - `paper_eval(df_train, df_test, out_dir) -> dict | None`
### Reporting only (`method == report`)

1. `cli.py` → `run_post_reporting(out_dir)`.
2. Rebuild tables/plots from existing `metrics.json` files under `out_dir`.
3. No model training or scoring is executed.

---

---

## Data flow

### Single run (CLI `method != all`)

1. `cli.py` → `run_single(...)` (in `eval.core`).
2. `load_train_test(...)` → `df_tr`, `df_te`.
3. `infer_dataset_label_from_df(df_te)`.
4. `_build_model(method, df_tr, log)` using `METHOD_REGISTRY`.
5. `model.fit(df_tr)`.
6. `score_windows` / `score_messages` if implemented.
7. `evaluate_scores_*` → metrics, ROC, confusion.
8. Optional `model.paper_eval(...)`.
9. Write `metrics.json`, ROC PNGs under `out_dir / dataset_label / method`.

### Full plan (`method == all`)

1. `cli.py` → `run_eval_plan(...)` (in `eval.parallel`).
2. For each `cfg` in `EVAL_PLAN`, create a **dataset job**.
3. `ProcessPoolExecutor` over dataset jobs (n_jobs processes).
4. Inside `_run_dataset_job`:
   - `load_train_test(...)` once.
   - For each `method`:
     - Skip if `metrics.json` exists.
     - Call `_run_single_on_loaded(...)`.
5. Collect results, write failure summary under `out_dir/_failures`.
6. Run post-reporting on `out_dir` (tables + plots) unless `--skip_report` is set.

