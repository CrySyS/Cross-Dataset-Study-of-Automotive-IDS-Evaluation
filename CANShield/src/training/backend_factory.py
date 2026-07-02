from training.keras_backend import KerasBackend
from training.torch_backend import TorchBackend


def get_model_backend(args):
    backend_name = str(getattr(args, "backend", "keras")).lower()
    if backend_name == "torch":
        return TorchBackend()
    return KerasBackend()
