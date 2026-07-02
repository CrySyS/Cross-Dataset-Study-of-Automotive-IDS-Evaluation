# convert_all.py
from pathlib import Path

from adapters.hcrl_car_hacking import convert_file as conv_hcrl_ch
from adapters.hcrl_otids import convert_file as conv_otids
# from adapters.can_train_test import convert_file as conv_cantt
from adapters.crysys_can import convert_folder as conv_crysys_folder
from adapters.syncan import convert_file as conv_syncan
from adapters.road import convert_file as conv_road
from adapters.can_intrusion_dataset_v2 import convert_folder as conv_can_intrusion_v2_folder

DATA_ROOT = Path("data_raw")
OUT_ROOT = Path("data_parquet")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

datasets = {
    "hcrl_car_hacking": "03_Car-HackingDataset",
    "hcrl_otids": "04_CAN-IntrusionDataset_OTIDS",
    "can_train_test": "",     # conv_cantt,
    "crysys": "06_CrySyS_dataset",
    "syncan": "01_SynCAN/csv_files",
    "road": "02_Road",        # RAW + signal files too
    "can_intrusion_dataset_v2": "07_CAN_Intrusion_Dataset_v2",
}

def main():
    # 1) HCRL Car-Hacking
    for f in (DATA_ROOT / datasets["hcrl_car_hacking"]).glob("*.csv"):
        output_path = OUT_ROOT / datasets["hcrl_car_hacking"] / (f.stem + ".parquet")
        # if parquet exists, skip
        if output_path.exists():
            print("HCRL Car-Hacking: Skipping existing", f)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("HCRL Car-Hacking:", conv_hcrl_ch(str(f), str(output_path)))

    # 2) SynCAN (signals)
    for f in (DATA_ROOT / datasets["syncan"]).glob("*.csv"):
        output_path = OUT_ROOT / datasets["syncan"] / (f.stem + ".signals.parquet")
        if output_path.exists():
            print("SynCAN: Skipping existing", f)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("SynCAN:", conv_syncan(str(f), str(output_path)))

    # 3) ROAD (raw CAN + signals)
    #   - walk all files under data_raw/02_Road
    #   - only consider *.log, *.txt, *.csv, *.zip
    #   - mirror the folder structure under data_parquet/02_Road/...
    #   - let conv_road decide .parquet vs .signal.parquet based on content
    road_root_in = DATA_ROOT / datasets["road"]
    road_root_out = OUT_ROOT / datasets["road"]

    if road_root_in.exists():
        for f in sorted(road_root_in.rglob("*")):
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf not in {".log", ".txt", ".csv", ".zip"}:
                continue

            # path inside 02_Road, e.g. "raw/attacks/foo.log"
            rel_in = f.relative_to(road_root_in)
            out_dir = road_root_out / rel_in.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Check if *either* frame or signal parquet already exists
            parquet_frame = out_dir / (f.stem + ".parquet")
            parquet_signal = out_dir / (f.stem + ".signal.parquet")
            if parquet_frame.exists() or parquet_signal.exists():
                print("ROAD: Skipping existing", f)
                continue

            # Pass the directory as path_out; conv_road will build filename with
            # .parquet or .signal.parquet based on schema.
            print("ROAD:", conv_road(str(f), str(out_dir)))
    else:
        print("ROAD: input root does not exist:", road_root_in)

    # 4) HCRL OTIDS
    dataset_name = "hcrl_otids"
    for f in (DATA_ROOT / datasets[dataset_name]).glob("*.txt"):
        output_path = OUT_ROOT / datasets[dataset_name] / (f.stem + ".parquet")
        if output_path.exists():
            print("HCRL OTIDS: Skipping existing", f)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("HCRL OTIDS:", conv_otids(str(f), str(output_path)))

    # 5) CrySyS SocketCAN logs
    crysys_root_in = DATA_ROOT / datasets["crysys"]
    crysys_root_out = OUT_ROOT / "06_CrySyS"
    
    if crysys_root_in.exists():
        print(f"Converting CrySyS dataset from {crysys_root_in}")
        output_paths = conv_crysys_folder(
            str(crysys_root_in),
            str(crysys_root_out),
            overwrite=False  # Skip existing files
        )
        print(f"CrySyS: Converted {len(output_paths)} files")
    else:
        print("CrySyS: input root does not exist:", crysys_root_in)

    # 6) CAN Intrusion Dataset v2 (OpelAstra / Prototype / RenaultClio)
    data_root_in = DATA_ROOT / datasets["can_intrusion_dataset_v2"]
    data_root_out = OUT_ROOT / "can_intrusion_dataset_v2"

    if data_root_in.exists():
        print(f"Converting CAN Intrusion Dataset v2 from {data_root_in}")
        output_paths = conv_can_intrusion_v2_folder(
            str(data_root_in),
            str(data_root_out),
            glob_pattern="*.log",
            dataset_name="CAN Intrusion Dataset v2",
            overwrite=False,  # Skip existing files
        )
        print(f"CAN Intrusion Dataset v2: Converted {len(output_paths)} files")
    else:
        print("CAN Intrusion Dataset v2: input root does not exist:", data_root_in)

if __name__ == "__main__":
    main()
