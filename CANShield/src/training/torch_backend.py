import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class TorchAutoencoder(nn.Module):
    def __init__(self, time_step, num_signals):
        super().__init__()
        self.time_step = time_step
        self.num_signals = num_signals

        self.pad = nn.ZeroPad2d((2, 2, 2, 2))

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.deconv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.dact1 = nn.LeakyReLU(negative_slope=0.2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.dact2 = nn.LeakyReLU(negative_slope=0.2)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.dact3 = nn.LeakyReLU(negative_slope=0.2)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.out_act = nn.Sigmoid()

    def _center_crop(self, tensor):
        _, _, height, width = tensor.shape
        top = max((height - self.time_step) // 2, 0)
        left = max((width - self.num_signals) // 2, 0)
        return tensor[:, :, top:top + self.time_step, left:left + self.num_signals]

    def forward(self, x):
        x = self.pad(x)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))

        x = self.up1(self.dact1(self.deconv1(x)))
        x = self.up2(self.dact2(self.deconv2(x)))
        x = self.up3(self.dact3(self.deconv3(x)))

        x = self.out_act(self.out_conv(x))
        x = self._center_crop(x)
        return x


class TorchBackend:
    name = "torch"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _model_path(self, args, sampling_period, ext="pt"):
        root_dir = args.root_dir
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        return Path(
            f"{root_dir}/../artifacts/models/{dataset_name}/"
            f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.{ext}"
        )

    def _checkpoint_path(self, args):
        root_dir = args.root_dir
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        sampling_period = args.sampling_period
        return Path(
            f"{root_dir}/../artifacts/model_ckpts/{dataset_name}/"
            f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.pt"
        )

    def get_autoencoder(self, args):
        root_dir = args.root_dir
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        sampling_period = args.sampling_period

        model_search_path = (
            f"{root_dir}/../artifacts/models/{dataset_name}/"
            f"autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_*.pt"
        )
        print(f"[DEBUG] Looking for models at: {model_search_path}")
        model_list = glob.glob(model_search_path)

        sp_best = 0
        best_model_dir = None
        for model_dir in model_list:
            file_name = Path(model_dir).name.split(".")[0]
            sp_existing = int(file_name.split("_")[-1])
            if sp_existing > sp_best and sp_existing <= sampling_period:
                sp_best = sp_existing
                best_model_dir = Path(model_dir)

        model = TorchAutoencoder(time_step, num_signals).to(self.device)
        retrain = True

        if best_model_dir is not None:
            state = torch.load(best_model_dir, map_location=self.device)
            model.load_state_dict(state)
            if sp_best == sampling_period:
                retrain = False
            print(f"Model loaded from {best_model_dir}")
        else:
            print("Model created...")

        return model, retrain

    def train_autoencoder(self, args, file_index, model, x_train_seq):
        dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))
        time_step = args.time_step
        num_signals = args.num_signals
        sampling_period = args.sampling_period
        max_epoch = args.max_epoch
        root_dir = args.root_dir

        print(f"Training on {'GPU' if self.device.type == 'cuda' else 'CPU'}")

        x_tensor = torch.from_numpy(x_train_seq.astype(np.float32)).permute(0, 3, 1, 2)
        full_dataset = TensorDataset(x_tensor, x_tensor)

        val_size = max(1, int(len(full_dataset) * 0.1))
        train_size = max(1, len(full_dataset) - val_size)
        if train_size + val_size > len(full_dataset):
            val_size = len(full_dataset) - train_size
        if val_size == 0:
            val_size = 1
            train_size = len(full_dataset) - 1

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.99))
        criterion = nn.MSELoss()

        checkpoint_path = self._checkpoint_path(args)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_state = None
        patience = 10
        wait = 0
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        for epoch in range(max_epoch):
            model.train()
            train_losses = []
            train_accs = []
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_acc = ((outputs >= 0.5) == (targets >= 0.5)).float().mean().item()
                train_accs.append(train_acc)

            model.eval()
            val_losses = []
            val_accs = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    val_losses.append(val_loss.item())
                    val_acc = ((outputs >= 0.5) == (targets >= 0.5)).float().mean().item()
                    val_accs.append(val_acc)

            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            train_acc = float(np.mean(train_accs)) if train_accs else 0.0
            val_acc = float(np.mean(val_accs)) if val_accs else 0.0
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{max_epoch} - accuracy: {train_acc:.4f} - loss: {train_loss:.6f} "
                f"- val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, checkpoint_path)
                print(f"val_loss improved to {val_loss:.6f}, saving model to {checkpoint_path}")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        history_dir = Path(
            f"{root_dir}/../artifacts/histories/{dataset_name}/"
            f"history_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index + 1}.json"
        )
        history_dir.parent.mkdir(exist_ok=True, parents=True)
        with open(history_dir, "w") as fp:
            json.dump(history, fp)

        return model

    def save_model(self, args, model):
        model_dir = self._model_path(args, args.sampling_period, ext="pt")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir)
        return model_dir

    def load_model_for_inference(self, args):
        model_dir = self._model_path(args, args.sampling_period, ext="pt")
        if not model_dir.exists():
            raise FileNotFoundError(f"Torch model not found: {model_dir}")

        model = TorchAutoencoder(args.time_step, args.num_signals).to(self.device)
        state = torch.load(model_dir, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def predict_reconstruction(self, model, x_seq):
        # Avoid moving the full sequence tensor to GPU at once; large files can explode VRAM usage.
        infer_batch_size = int(os.environ.get("CANSHIELD_TORCH_INFER_BATCH_SIZE", "256"))
        x_np = x_seq.astype(np.float32, copy=False)
        recon_chunks = []

        with torch.no_grad():
            for start in range(0, x_np.shape[0], infer_batch_size):
                end = min(start + infer_batch_size, x_np.shape[0])
                x_tensor = torch.from_numpy(x_np[start:end]).permute(0, 3, 1, 2).to(self.device)
                recon_chunk = model(x_tensor).permute(0, 2, 3, 1).cpu().numpy()
                recon_chunks.append(recon_chunk)

                # Release per-batch GPU tensors before next chunk.
                del x_tensor
                del recon_chunk

        if len(recon_chunks) == 1:
            return recon_chunks[0]
        return np.concatenate(recon_chunks, axis=0)
