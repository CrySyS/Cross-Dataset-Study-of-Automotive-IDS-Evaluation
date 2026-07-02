from tqdm import tqdm
import hydra
from omegaconf import DictConfig, open_dict
import gc
import numpy as np
from pathlib import Path
from dataset.load_dataset import *
# from hydra.utils import get_original_cwd
from training import *


@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def develop_canshield(args : DictConfig) -> None:
    root_dir = Path(__file__).resolve().parent
    print("root_dir: ", root_dir)
    args.root_dir = root_dir
    args.data_type = "training"
    args.data_dir = args.train_data_dir
    print("Current working dir: ", args.root_dir)
    dataset_name = args.dataset_name
    num_signals = args.num_signals
    debug_outputs = bool(args.get("debug_outputs", False)) or bool(args.get("debug_input_pipeline", False))
    debug_output_suffix = str(args.get("debug_output_suffix", "debug")).strip()
    explicit_output_dataset_name = str(args.get("output_dataset_name", "")).strip()
    with open_dict(args):
        if explicit_output_dataset_name:
            args.output_dataset_name = explicit_output_dataset_name
            print(
                "Using explicit output dataset namespace "
                f"'{args.output_dataset_name}'"
            )
        elif debug_outputs:
            suffix = debug_output_suffix if debug_output_suffix else "debug"
            args.output_dataset_name = f"{dataset_name}_{suffix}"
            print(
                "Debug outputs enabled: writing artifacts under "
                f"dataset namespace '{args.output_dataset_name}'"
            )
        else:
            args.output_dataset_name = dataset_name

    backend = get_model_backend(args)
    debug_limit_files = int(args.get("debug_max_files", 0))

    def _summarize_tensor(name, tensor):
        print(
            f"[{name}] shape={tensor.shape}, dtype={tensor.dtype}, "
            f"min={float(np.min(tensor)):.6f}, max={float(np.max(tensor)):.6f}, "
            f"mean={float(np.mean(tensor)):.6f}, std={float(np.std(tensor)):.6f}, "
            f"nan={int(np.isnan(tensor).sum())}, inf={int(np.isinf(tensor).sum())}"
        )

    for time_step in args.time_steps:
        for sampling_period in args.sampling_periods:
            # Sep-up variable to define the AE model
            args.time_step = time_step
            args.sampling_period = sampling_period
            args.window_step = args.window_step_train
            print(f"Starting thresholding with args.window_step: {args.window_step}")

            # Train individual AE for each combination
            autoencoder, retrain = backend.get_autoencoder(args)

            if retrain is False:
                print("Model already trained for this setting. Skipping retraining.")
                continue
            
            file_dir_dict = get_list_of_files(args)
            print("file_dir_dict: ", file_dir_dict)
            assert len(file_dir_dict) > 0, "No files found in the specified directory."
            for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):
                try:
                    if debug_limit_files > 0 and file_index >= debug_limit_files:
                        print(f"Stopping after debug_max_files={debug_limit_files} files for this run.")
                        break

                    print("Starting loading", file_index, file_name)
                    x_train_seq, _ = load_data_create_images(args, file_name, file_path)
                    _summarize_tensor(f"train_input::{file_name}", x_train_seq)
                
                    print("Starting trainin with", file_name)
                    autoencoder = backend.train_autoencoder(args, file_index, autoencoder, x_train_seq)
                    del x_train_seq
                    gc.collect()
                except Exception as error:
                    print(error)
                    print(f"Skipping dataset {file_name}")

            model_dir = backend.save_model(args, autoencoder)
            print("Model saved.......!")

print("Training AE Models for CANShield is complete!")

if __name__ == "__main__":
    develop_canshield()