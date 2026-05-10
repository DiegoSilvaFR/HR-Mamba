import os
import argparse
import random
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score
from models.hr_mamba import HRMambaRegressor

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)

def compute_ENMO(X):
    
    X = np.asarray(X)

    # X shape expected: (n, 3, time)
    enmo_raw = np.sqrt(X[:, 0, :]**2 + X[:, 1, :]**2 + X[:, 2, :]**2).mean(axis=1)

    enmo = np.maximum(enmo_raw - 1, 0)

    return 1000 * enmo
# -----------------------------
# Dataset
# -----------------------------
class HRDataset(Dataset):
    """
    Each row in file_list.csv points to one subject-level .npz file with:
      - X: float32 array [n_windows, 3, T]
      - hr: float32 array [n_windows]

    We sample a fixed number of windows per subject at each __getitem__,
    following Oxford's subject-balanced batching idea.
    """

    def __init__(
        self,
        file_list_path: str,
        num_sample_per_subject: int,
        ratio2keep: float = 1.0,
        weighted_sample: bool = False,
        hr_min: Optional[float] = None,
        hr_max: Optional[float] = None,
        is_hr_classification: bool = False,
        hr_bins: Optional[np.ndarray] = None,
        expected_t: Optional[int] = None,
        is_enmo_regression: bool = False
    ) -> None:
        df = pd.read_csv(file_list_path)

        if "file_list" in df.columns:
            self.file_list = df["file_list"].tolist()
        elif "path" in df.columns:
            self.file_list = df["path"].tolist()
        else:
            raise ValueError("file_list.csv must contain a 'file_list' or 'path' column.")

        self.num_sample_per_subject = num_sample_per_subject
        self.ratio2keep = ratio2keep
        self.weighted_sample = weighted_sample
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.is_hr_classification = is_hr_classification
        self.hr_bins = None if hr_bins is None else np.asarray(hr_bins, dtype=np.float32)
        self.expected_t = expected_t
        self.is_enmo_regression = is_enmo_regression

    def __len__(self) -> int:
        return len(self.file_list)

    def _sample_indices(self, X: np.ndarray, hr: np.ndarray) -> np.ndarray:
        n = len(X)
        if n < self.num_sample_per_subject:
            raise ValueError(
                f"Subject has only {n} windows, but num_sample_per_subject={self.num_sample_per_subject}."
            )

        if not self.weighted_sample:
            return np.random.choice(n, size=self.num_sample_per_subject, replace=False)

        # HR-bin balanced weighted sampling:
        # each non-empty bin gets approximately the same total sampling mass.
        if self.is_enmo_regression:
            bins = np.array(
            [40, 100], dtype=np.float32
        )
        else:
            bins = self.hr_bins if self.hr_bins is not None else np.array(
                [40, 55, 70, 85, 100, 120, 140, 180, 220], dtype=np.float32
            )

        hr_bin = np.digitize(hr, bins=bins, right=False)
        unique_bins, counts = np.unique(hr_bin, return_counts=True)
        count_map = dict(zip(unique_bins, counts))

        weights = np.array([1.0 / count_map[b] for b in hr_bin], dtype=np.float64)
        weights = np.maximum(weights, 1e-12)
        p = weights / weights.sum()

        return np.random.choice(n, size=self.num_sample_per_subject, replace=False, p=p)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        data = np.load(path, allow_pickle=True)

        X = data["X"].astype(np.float32)   # [N, 3, T]
        if self.is_enmo_regression is False:
            hr = data["hr"].astype(np.float32) # [N]
        else:
            hr = compute_ENMO(X)

        if X.ndim != 3:
            raise ValueError(f"Expected X.ndim == 3 in {path}, got {X.shape}")

        # Safety: fix [N, T, 3] -> [N, 3, T] if needed
        if X.shape[1] != 3 and X.shape[2] == 3:
            X = np.transpose(X, (0, 2, 1))

        if X.shape[1] != 3:
            raise ValueError(f"Expected X.shape[1] == 3 in {path}, got {X.shape}")

        if self.expected_t is not None and X.shape[2] != self.expected_t:
            raise ValueError(
                f"Unexpected temporal length in {path}. Expected T={self.expected_t}, got {X.shape[2]}"
            )

        keep_n = int(len(X) * self.ratio2keep)
        keep_n = max(keep_n, self.num_sample_per_subject)

        X = X[:keep_n]
        hr = hr[:keep_n]

        if self.hr_min is not None:
            valid = hr >= self.hr_min
            X, hr = X[valid], hr[valid]

        if self.hr_max is not None:
            valid = hr <= self.hr_max
            X, hr = X[valid], hr[valid]

        sel = self._sample_indices(X, hr)

        X = torch.from_numpy(X[sel]).float().contiguous()

        if self.is_hr_classification:
            if self.hr_bins is None:
                raise ValueError("For classification, hr_bins must be provided.")
            hr_classes = np.digitize(hr, self.hr_bins)
            hr = torch.from_numpy(hr_classes[sel]).long().contiguous()
        else:
            hr = torch.from_numpy(hr[sel]).float().contiguous()

        return X, hr


def hr_subject_collate(batch):
    X = torch.cat([item[0] for item in batch], dim=0).contiguous()   # [B*S, 3, T]
    hr = torch.cat([item[1] for item in batch], dim=0).contiguous()  # [B*S]
    return X, hr


# -----------------------------
# Metrics / loops
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, loss_fn, is_hr_classification: bool = False):
    model.eval()

    losses = []
    y_true = []
    y_pred = []

    for X, hr in loader:
        X = X.to(device, dtype=torch.float32).contiguous()

        if is_hr_classification:
            hr = hr.to(device, dtype=torch.long)
            pred = model(X)
            loss = loss_fn(pred, hr)
            pred_class = torch.argmax(pred, dim=1)

            losses.append(loss.item())
            y_true.append(hr.cpu())
            y_pred.append(pred_class.cpu())
        else:
            hr = hr.to(device, dtype=torch.float32)
            pred = model(X).reshape(-1)
            loss = loss_fn(pred, hr)

            losses.append(loss.item())
            y_true.append(hr.cpu())
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if is_hr_classification:
        acc = float((y_true == y_pred).mean())
        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return {
            "loss": float(np.mean(losses)),
            "acc": acc,
            "precision": precision,
            "recall": recall,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "rmse": rmse,
    }


def train_one_epoch(model, loader, optimizer, device, loss_fn, is_hr_classification: bool):
    model.train()

    losses = []
    y_true = []
    y_pred = []

    for X, hr in loader:
        X = X.to(device, dtype=torch.float32).contiguous()

        if is_hr_classification:
            hr = hr.to(device, dtype=torch.long)
            pred = model(X)
            loss = loss_fn(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_class = torch.argmax(pred, dim=1)
            losses.append(loss.item())
            y_true.append(hr.detach().cpu())
            y_pred.append(pred_class.detach().cpu())
        else:
            hr = hr.to(device, dtype=torch.float32)
            pred = model(X).reshape(-1)
            loss = loss_fn(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            y_true.append(hr.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if is_hr_classification:
        acc = float((y_true == y_pred).mean())
        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return {
            "loss": float(np.mean(losses)),
            "acc": acc,
            "precision": precision,
            "recall": recall,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "rmse": rmse,
    }


# -----------------------------
# Model helpers
# -----------------------------

def infer_expected_t(epoch_len: int) -> int:
    mapping = {5: 150, 10: 300, 30: 900}
    if epoch_len not in mapping:
        raise ValueError(f"Unsupported epoch_len={epoch_len}. Use 5, 10 or 30.")
    return mapping[epoch_len]

def load_backbone_only_from_checkpoint(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    backbone_state = {
        k: v for k, v in clean_state.items()
        if k.startswith("feature_extractor.")
    }

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"Loaded backbone from checkpoint: {ckpt_path}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def get_head_parameters(model):
    if hasattr(model, "classifier"):
        return list(model.classifier.parameters())
    raise ValueError("Model has no attribute 'classifier'.")



# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file-list", type=str, required=True)
    parser.add_argument("--test-file-list", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--batch-subject-num", type=int, default=4)
    parser.add_argument("--num-sample-per-subject", type=int, default=250)
    parser.add_argument("--num-sample-test", type=int, default=100)

    parser.add_argument("--weighted-sample", action="store_true")
    parser.add_argument("--enmo-regression", action="store_true")

    parser.add_argument("--epoch-len", type=int, default=10, choices=[5, 10, 30])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--version", type=int, default=1)
    # task mode
    parser.add_argument("--hr-classification", action="store_true")
    parser.add_argument(
        "--hr-bins",
        type=str,
        default="100",
        help="Comma-separated thresholds for HR classification, e.g. '100' or '80,100' or '50,65,80,95,110'",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    is_hr_classification = args.hr_classification
    is_enmo_regression = args.enmo_regression
    main_target = 'ENMO' if is_enmo_regression else 'HR'
    
    expected_t = infer_expected_t(args.epoch_len)

    if is_hr_classification:
        hr_bins = np.array([float(x) for x in args.hr_bins.split(",") if x.strip() != ""], dtype=np.float32)
      
        suffix = "cls"
    else:
        hr_bins = None
   
        suffix = "reg"

    train_ds = HRDataset(
        file_list_path=args.train_file_list,
        num_sample_per_subject=args.num_sample_per_subject,
        ratio2keep=args.ratio2keep,
        weighted_sample=args.weighted_sample,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        is_hr_classification=is_hr_classification,
        hr_bins=hr_bins,
        expected_t=expected_t,
        is_enmo_regression= is_enmo_regression
    )

    test_ds = HRDataset(
        file_list_path=args.test_file_list,
        num_sample_per_subject=args.num_sample_test,
        ratio2keep=1.0,
        weighted_sample=False,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        is_hr_classification=is_hr_classification,
        hr_bins=hr_bins,
        expected_t=expected_t,
        is_enmo_regression= is_enmo_regression
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_subject_num,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=hr_subject_collate,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_subject_num,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=hr_subject_collate,
        worker_init_fn=worker_init_fn,
    )

    model = HRMambaRegressor(
    input_channels=3,
    seq_len=300,
    patch_size=15,
    stride=15,
    d_model=128,
    depth=4,
    d_state=16,
)
    model = model.to(device)
    # optimizer = build_optimizer(
    #     model=model,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     freeze_backbone=args.freeze_backbone,
    #     backbone_lr_mult=args.backbone_lr_mult,
    # )

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

    loss_fn = nn.CrossEntropyLoss() if is_hr_classification else nn.MSELoss()

    init_suffix = 'MAMBA'
    best_path = os.path.join(args.save_dir, f"{main_target}_{suffix}_{init_suffix}_v{args.version}.mdl")
    hist_path = os.path.join(args.save_dir, f"{main_target}_evolution_parameters_v{args.version}_{suffix}_{init_suffix}.json")

    # early_stopping = EarlyStopping(
    #     patience=args.patience,
    #     path=best_path,
    #     verbose=True,
    # )

    evolution_parameters = {}

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, is_hr_classification
        )
        test_metrics = evaluate(
            model, test_loader, device, loss_fn, is_hr_classification
        )

        if is_hr_classification:
            msg = (
                f"[{epoch+1:03d}/{args.epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} "
                f"train_precision={train_metrics['precision']:.4f} "
                f"train_recall={train_metrics['recall']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"test_precision={test_metrics['precision']:.4f} "
                f"test_recall={test_metrics['recall']:.4f}"
            )
            evolution_parameters[epoch] = {
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc": round(train_metrics["acc"], 4),
                "train_precision": round(train_metrics["precision"], 4),
                "train_recall": round(train_metrics["recall"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "test_acc": round(test_metrics["acc"], 4),
                "test_precision": round(test_metrics["precision"], 4),
                "test_recall": round(test_metrics["recall"], 4),
            }
        else:
            msg = (
                f"[{epoch+1:03d}/{args.epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_mae={train_metrics['mae']:.4f} "
                f"train_rmse={train_metrics['rmse']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_mae={test_metrics['mae']:.4f} "
                f"test_rmse={test_metrics['rmse']:.4f}"
            )
            evolution_parameters[epoch] = {
                "train_loss": round(train_metrics["loss"], 4),
                "train_mae": round(train_metrics["mae"], 4),
                "train_rmse": round(train_metrics["rmse"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "test_mae": round(test_metrics["mae"], 4),
                "test_rmse": round(test_metrics["rmse"], 4),
            }

        print(msg)

        # early_stopping(test_metrics["loss"], model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    print(f"Best checkpoint saved to: {best_path}")

    with open(hist_path, "w") as file:
        json.dump(evolution_parameters, file, indent=4)


if __name__ == "__main__":
    main()
