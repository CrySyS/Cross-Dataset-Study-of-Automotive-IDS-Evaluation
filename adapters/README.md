

# Adapters

These adapters **ingest heterogeneous CAN / signal datasets and emit a unified format** so the rest of the pipeline (EDA, modeling, benchmarking) can treat them consistently.

They **do not download** datasets; they only **read local files and write canonical Parquet**.

## What an adapter does (and doesn’t)

**Does**

* Parse a dataset’s native files (CSV, TXT/SocketCAN logs, ZIP-of-CSV, etc.).
* Normalize fields and **output a canonical table**:

  * **Frame-space** (raw CAN frames), or
  * **Signal-space** (per-ID time series).
* Add lightweight **derived fields** (e.g., inter-arrival times) and **metadata** (dataset name, attack_type, split).
* Perform **non-fatal validation** (e.g., expected columns, label invariants) and warn if something is off.

**Doesn’t**

* Download or license-check datasets.
* Relabel data (except when a dataset only gives attack **windows**; then labels are assigned by time window, see OTIDS notes).
* Impute or “fix” data beyond harmless normalization (e.g., padding data bytes to 8).

---

## Canonical schemas

### 1) Frame-space (raw CAN) — Parquet

Used by: HCRL Car-Hacking, OTIDS (frames), ROAD (raw), CrySyS, etc.

| column           | type            | notes                                                      |
| ---------------- | --------------- | ---------------------------------------------------------- |
| `timestamp`      | float           | seconds (origin = file’s native origin, not absolute time) |
| `can_id`         | int64           | 11/29-bit; hex in source → int here                        |
| `dlc`            | int8            | 0..8; **actual** byte count from original frame            |
| `inter_arrival`  | float           | seconds between frames (per file ordering)                 |
| `data0`..`data7` | uint8           | payload bytes; zero-padded if `dlc<8` (use dlc to know valid bytes) |
| `label`          | int8            | 0=benign, 1=attack (from dataset or attack windows)        |
| `attack_type`    | string/nullable | e.g., `DoS`, `Fuzzy`, `Masquerade`                         |
| `dataset`        | string          | e.g., `HCRL-CarHacking`, `ROAD`                            |
| `frame_type`     | string/nullable | e.g., `remote_req`, `response`, `normal`                   |
| `vehicle`        | string/nullable | if available                                               |
| `split`          | string/nullable | train/test/val if available                                |
| `idx_src`        | int64           | 0..N-1 line index in source file                           |

### 2) Signal-space (per-ID time series) — Parquet

Used by: SynCAN, ROAD (optional signal CSVs), any decoded-signal sources.

**Note:** Signal-space now uses **unified column names** matching frame-space for consistency.

| column               | type            | notes                                     |
| -------------------- | --------------- | ----------------------------------------- |
| `timestamp`          | float           | seconds (unified with frame-space)        |
| `can_id`             | int64           | CAN ID as integer (unified with frame-space) |
| `label`              | int8            | 0/1 from dataset (SynCAN train are all 0) |
| `dataset`            | string          | e.g., `SynCAN`, `ROAD`                    |
| `attack_type`        | string/nullable | inferred from filename if applicable      |
| `split`              | string/nullable | train/test/val if available               |
| `idx_src`            | int64           | 0..N-1                                    |
| `signal1`..`signalN` | float/nullable  | decoded signal values (count varies by ID) |
| `inter_arrival`      | float           | seconds between messages (unified with frame-space) |

> **Why two schemas?** Some datasets only provide **raw frames**; others (like SynCAN) provide **signals** only. Keeping both avoids inventing fake frames or fake signals.
> 
> **Column name unification (2026-01-04):** Signal-space now uses the same column names as frame-space (`timestamp`, `can_id`, `inter_arrival`) instead of the old names (`timestamp_s`, `id_str`, `inter_arrival_s`). This simplifies code that needs to work with both formats.

---

## Current adapters

* `hcrl_car_hacking.py` — HCRL Car-Hacking (per-frame flag → label, attack_type inferred from filename).
* `hcrl_otids.py` — OTIDS / CAN-intrusion (supports **attack windows** and remote request markers if present).
* `road.py` — ROAD (SocketCAN logs → frames; optional signal CSVs → signals; flags `is_filler`, sets `obfuscated=True`).
* `crysys_can.py` — CrySyS (SocketCAN logs → frames; label set by scenario).
* `can_train_test.py` — CAN-Train-and-Test (labeled CSVs; preserves `vehicle` and `split`).
* `syncan.py` — SynCAN (ZIP/CSV → **signals**; honors provided `Label`; supports train concatenation).
* `can_intrusion_dataset_v2.py` — CAN Intrusion Dataset v2 candump-style logs under `data_raw/data` (OpelAstra / Prototype / RenaultClio) → frames; infers `vehicle`, `split`, and attack labels via README-defined windows/rules.

All adapters **only** emit canonical Parquet files (same column order per schema).

---

## Folder layout

```
project/
  adapters/
    README.md
    *.py
  data_raw/
    hcrl_car_hacking/*.csv
    hcrl_otids/*.csv
    road/raw/*.log
    road/signals/*.csv
    crysys/**/*.log
    can_train_test/*.csv
    syncan/*.zip, *.csv
    data/{OpelAstra,Prototype,RenaultClio}/*.log
  data_parquet/
    ...
  convert_all.py
```

---

## How to run

1. Install deps:

```bash
pip install pandas pyarrow numpy
```

2. Put raw files under `data_raw/...` as above.

3. Use the provided driver:

```bash
python convert_all.py
```

It writes Parquet files under `data_parquet/`, matching input filenames. Example outputs:

* `data_parquet/hcrl_car_hacking/DoS_01.parquet` (frames)
* `data_parquet/road/drive01.parquet` (frames)
* `data_parquet/road/drive01.signals.parquet` (signals)
* `data_parquet/syncan/train_all.signals.parquet` (signals, concatenated)

---

## Conventions & guarantees

* **Timestamps:** kept in **seconds**; relative to file origin (no global epoch alignment). Both frame-space and signal-space use `timestamp` column.
* **IDs:** hex → int (`can_id`) for both frames and signals (unified as of 2026-01-04).
* **DLC:** preserves **actual** byte count from original frame (not padded to 8).
* **Payload:** `data0..data7` as uint8, zero-padded if DLC<8 (use `dlc` field to determine valid bytes); frame-space only.
* **Labels:** adapters **trust dataset labels**. For datasets with only **attack windows**, labels are assigned by time interval (documented in code).
* **No resampling:** we do not resample or aggregate; that’s left to downstream code.
* **Non-fatal validation:** adapters may print warnings (unexpected IDs, non-monotonic time, etc.) but still write outputs.

---

## Adding a new adapter (checklist)

1. **Decide schema:** frames or signals. Don’t mix in one file.
2. **Parse:** implement a `load_*` that reads native files into a `DataFrame`.
3. **Normalize:** implement `to_canonical_*()` that:

   * renames columns,
   * converts types (timestamp → seconds, hex IDs → int),
   * creates required canonical columns,
   * computes `inter_arrival` (unified name for both frame and signal space).
4. **Metadata:** fill `dataset`, `attack_type`, `split`, `vehicle` if available.
5. **Write:** `convert_file(input_path, output_path=None)` → writes Parquet.
6. **(Optional) Validate:** add small checks; print warnings, don’t fail unless unrecoverable.
7. **Register in driver:** import and call from `convert_all.py`.

Use existing adapters as templates.

---


## Minimal code contracts (per adapter)

Each adapter should expose at least:

```python
def convert_file(path_in: str, path_out: str | None = None) -> str:
    """Read native file(s), write canonical Parquet, return output path."""
```

Optionally:

```python
def convert_folder(root_in: str, out_dir: str | None = None) -> list[str]:
    """Batch convert a folder tree; return list of outputs."""
```


---

# Notes on optional fields

- can_id_hex: Original hexadecimal string ID, optional (useful for debugging or matching manufacturer documentation).

- data_len: Number of payload bytes (≤ 8); useful when datasets include remote frames or shorter DLC.

- timestamp_ns: Derived from timestamp × 1e9 for high-resolution sorting; optional.

- frame_type: Optional categorical (normal, remote_req, response).

- flag / direction: Optional field from datasets that record transmit/receive direction (e.g., HCRL Car-Hacking).

Datasets lacking these fields should fill None or zero defaults. Downstream code must not assume their presence.