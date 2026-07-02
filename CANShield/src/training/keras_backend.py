from pathlib import Path
from keras.models import load_model

from training.get_autoencoder import get_autoencoder
from training.train_autoencoder import train_autoencoder


class KerasBackend:
    name = "keras"

    def get_autoencoder(self, args):
        return get_autoencoder(args)

    def train_autoencoder(self, args, file_index, model, x_train_seq):
        return train_autoencoder(args, file_index, model, x_train_seq)

    def save_model(self, args, model):
        root_dir = args.root_dir
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        sampling_period = args.sampling_period
        model_dir = Path(
            f"{root_dir}/../artifacts/models/{dataset_name}/"
            f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.keras"
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_dir)
        return model_dir

    def load_model_for_inference(self, args):
        root_dir = args.root_dir
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        sampling_period = args.sampling_period

        model_dir = Path(
            f"{root_dir}/../artifacts/models/{dataset_name}/"
            f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.keras"
        )
        if not model_dir.exists():
            model_dir = Path(
                f"{root_dir}/../artifacts/models/{dataset_name}/"
                f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
            )
        return load_model(model_dir)

    def predict_reconstruction(self, model, x_seq):
        return model.predict(x_seq)
