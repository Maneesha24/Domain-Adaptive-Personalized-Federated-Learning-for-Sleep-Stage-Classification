"""
Domain-Adaptive Personalized Federated Learning for Sleep Stage Classification
----------------------------------------------------------------------------

Final experiment script for the paper.

Runs:
1. Centralized CNN-BiLSTM upper bound
2. FedAvg
3. Personalized FL = FedAvg + client-specific fine-tuning
4. DA-PFL = GRL-based domain-adaptive FL + fixed Phase-2 personalization

Key paper-aligned details:
- Sleep-EDF Cassette, 10 subjects x 2 nights = 20 simulated clients
- Each recording is treated as one federated client
- Phase 1 DA-PFL uses classification + domain loss with GRL
- Phase 2 DA-PFL uses classification loss only; no domain loss
- Multi-seed evaluation with mean ± std
- Saves CSV results, LaTeX table rows, convergence curves, and per-class reports

Usage:
    python index.py

Optional:
    python index.py --seeds 42 52 62
    python index.py --fed-rounds 10
    python index.py --skip-centralized

Dependencies:
    pip install torch torchvision torchaudio mne scikit-learn numpy pandas tqdm
"""

import argparse
import copy
import json
import os
import random
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Defaults
# =============================================================================

SUBJECTS = list(range(10))
RECORDS = [1, 2]
CHANNEL = "EEG Fpz-Cz"
EPOCH_SEC = 30
SFREQ = 100

BATCH_SIZE = 64
CENTRAL_EPOCHS = 10
FED_ROUNDS = 10
LOCAL_EPOCHS = 1
PERSONAL_EPOCHS = 3

LR = 1e-3
PERSONAL_LR_FACTOR = 0.1
DOMAIN_LAMBDA = 0.1

DEFAULT_SEEDS = [42, 52, 62]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANNOTATION_DESC_2_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

ID_2_LABEL = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# =============================================================================
# Gradient Reversal Layer
# =============================================================================

class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lam)


# =============================================================================
# Dataset loading
# =============================================================================

def load_sleep_edf(subjects=SUBJECTS, records=RECORDS):
    """Download/load Sleep-EDF and return X, y, client_ids."""
    from mne.datasets.sleep_physionet.age import fetch_data

    X_all, y_all, client_ids = [], [], []
    files = fetch_data(subjects=subjects, recording=records)

    for client_idx, (psg_file, hyp_file) in enumerate(files):
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_file)

        raw.set_annotations(annot, emit_warning=False)
        raw.pick([CHANNEL])
        raw.filter(0.3, 35.0, fir_design="firwin", verbose=False)
        raw.resample(SFREQ, verbose=False)

        events, event_id = mne.events_from_annotations(
            raw,
            event_id=ANNOTATION_DESC_2_ID,
            chunk_duration=EPOCH_SEC,
            verbose=False,
        )

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=EPOCH_SEC - 1.0 / SFREQ,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = epochs.get_data()  # shape: N x 1 x 3000
        y = epochs.events[:, -1]

        # Per-epoch z-score normalization.
        X = (X - X.mean(axis=2, keepdims=True)) / (
            X.std(axis=2, keepdims=True) + 1e-8
        )

        X_all.append(X.astype(np.float32))
        y_all.append(y.astype(np.int64))
        client_ids.extend([client_idx] * len(y))

    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.array(client_ids),
    )


class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_clients(X, y, client_ids, seed: int, test_size: float = 0.2):
    """Stratified 80/20 train-test split per client."""
    clients = {}

    for cid in np.unique(client_ids):
        idx = np.where(client_ids == cid)[0]

        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=y[idx],
        )

        clients[int(cid)] = {
            "train": SleepDataset(X[train_idx], y[train_idx]),
            "test": SleepDataset(X[test_idx], y[test_idx]),
            "num_train": len(train_idx),
            "num_test": len(test_idx),
        }

    return clients


def centralized_split(X, y, seed: int, test_size: float = 0.2):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return train_idx, test_idx


# =============================================================================
# Models
# =============================================================================

class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x):
        h = self.extract_features(x)
        return self.classifier(h)


class DomainAdaptiveCNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 5, num_domains: int = 20, lam: float = 0.1):
        super().__init__()

        self.backbone = CNNBiLSTM(num_classes=num_classes)
        self.grl = GradientReversalLayer(lam=lam)

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_domains),
        )

    def forward(self, x):
        h = self.backbone.extract_features(x)
        class_logits = self.backbone.classifier(h)
        domain_logits = self.domain_classifier(self.grl(h))
        return class_logits, domain_logits


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_classifier(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            logits = model(X_b)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y_b.numpy())
            y_pred.extend(preds)

    return metrics_from_predictions(y_true, y_pred)


def evaluate_domain_model(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            class_logits, _ = model(X_b)
            preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            y_true.extend(y_b.numpy())
            y_pred.extend(preds)

    return metrics_from_predictions(y_true, y_pred)


def metrics_from_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kappa, list(y_true), list(y_pred)


def aggregate_global_eval(model, clients, domain_model: bool = False):
    all_true, all_pred = [], []

    for data in clients.values():
        loader = DataLoader(data["test"], batch_size=BATCH_SIZE, shuffle=False)
        if domain_model:
            _, _, _, yt, yp = evaluate_domain_model(model, loader)
        else:
            _, _, _, yt, yp = evaluate_classifier(model, loader)
        all_true.extend(yt)
        all_pred.extend(yp)

    acc, f1, kappa, _, _ = metrics_from_predictions(all_true, all_pred)
    return acc, f1, kappa, all_true, all_pred


def evaluate_personalized(models, clients, domain_model: bool = False):
    all_true, all_pred = [], []

    for cid, model in models.items():
        loader = DataLoader(clients[cid]["test"], batch_size=BATCH_SIZE, shuffle=False)

        if domain_model:
            _, _, _, yt, yp = evaluate_domain_model(model, loader)
        else:
            _, _, _, yt, yp = evaluate_classifier(model, loader)

        all_true.extend(yt)
        all_pred.extend(yp)

    acc, f1, kappa, _, _ = metrics_from_predictions(all_true, all_pred)
    return acc, f1, kappa, all_true, all_pred


def save_classification_report(y_true, y_pred, out_path: Path):
    target_names = [ID_2_LABEL[i] for i in range(5)]
    text_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    dict_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    out_path.write_text(text_report)
    pd.DataFrame(dict_report).transpose().to_csv(out_path.with_suffix(".csv"))


# =============================================================================
# Training helpers
# =============================================================================

def train_one_epoch_classifier(model, loader, optimizer, criterion, desc="Training"):
    model.train()
    total_loss = 0.0

    for X_b, y_b in tqdm(loader, desc=desc, leave=False):
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def fedavg(state_dicts, client_sizes):
    avg_state = copy.deepcopy(state_dicts[0])
    total = float(sum(client_sizes))

    for key in avg_state:
        avg_state[key] = sum(
            state_dicts[i][key] * (client_sizes[i] / total)
            for i in range(len(state_dicts))
        )

    return avg_state


# =============================================================================
# Centralized baseline
# =============================================================================

def run_centralized(X, y, seed: int, out_dir: Path, verbose: bool = False):
    set_seed(seed)

    train_idx, test_idx = centralized_split(X, y, seed=seed)

    train_loader = DataLoader(
        SleepDataset(X[train_idx], y[train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        SleepDataset(X[test_idx], y[test_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNNBiLSTM(num_classes=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(CENTRAL_EPOCHS):
        loss = train_one_epoch_classifier(
            model,
            train_loader,
            optimizer,
            criterion,
            desc=f"Central seed={seed} ep={epoch+1}",
        )
        if verbose:
            acc, f1, kappa, _, _ = evaluate_classifier(model, test_loader)
            print(
                f"Centralized | seed={seed} | epoch={epoch+1}/{CENTRAL_EPOCHS} "
                f"| loss={loss:.4f} | acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f}"
            )

    acc, f1, kappa, y_true, y_pred = evaluate_classifier(model, test_loader)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "centralized.pt")
    save_classification_report(y_true, y_pred, seed_dir / "centralized_report.txt")

    return {
        "method": "Centralized CNN-BiLSTM",
        "seed": seed,
        "accuracy": acc,
        "macro_f1": f1,
        "kappa": kappa,
    }


# =============================================================================
# FedAvg
# =============================================================================

def train_local_fedavg(global_model, train_loader):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        train_one_epoch_classifier(
            local_model,
            train_loader,
            optimizer,
            criterion,
            desc="Local FedAvg",
        )

    return local_model.state_dict()


def run_fedavg(clients, seed: int, out_dir: Path, verbose: bool = False):
    set_seed(seed)

    global_model = CNNBiLSTM(num_classes=5).to(DEVICE)
    convergence = []

    for rnd in range(FED_ROUNDS):
        local_states, client_sizes = [], []

        for data in clients.values():
            loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
            local_states.append(train_local_fedavg(global_model, loader))
            client_sizes.append(data["num_train"])

        global_model.load_state_dict(fedavg(local_states, client_sizes))

        acc, f1, kappa, _, _ = aggregate_global_eval(
            global_model,
            clients,
            domain_model=False,
        )

        convergence.append({
            "method": "FedAvg",
            "seed": seed,
            "round": rnd + 1,
            "accuracy": acc,
            "macro_f1": f1,
            "kappa": kappa,
        })

        if verbose:
            print(
                f"FedAvg | seed={seed} | round={rnd+1}/{FED_ROUNDS} "
                f"| acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f}"
            )

    acc, f1, kappa, y_true, y_pred = aggregate_global_eval(
        global_model,
        clients,
        domain_model=False,
    )

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), seed_dir / "fedavg.pt")
    save_classification_report(y_true, y_pred, seed_dir / "fedavg_report.txt")

    return global_model, {
        "method": "FedAvg",
        "seed": seed,
        "accuracy": acc,
        "macro_f1": f1,
        "kappa": kappa,
    }, convergence


# =============================================================================
# Personalized FL
# =============================================================================

def personalize_fedavg(global_model, clients):
    personalized = {}

    for cid, data in clients.items():
        model = copy.deepcopy(global_model).to(DEVICE)
        loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR * PERSONAL_LR_FACTOR)
        criterion = nn.CrossEntropyLoss()

        for ep in range(PERSONAL_EPOCHS):
            train_one_epoch_classifier(
                model,
                loader,
                optimizer,
                criterion,
                desc=f"PersFL cid={cid} ep={ep+1}",
            )

        personalized[cid] = model

    return personalized


def run_personalized_fl(fedavg_model, clients, seed: int, out_dir: Path):
    models = personalize_fedavg(fedavg_model, clients)
    acc, f1, kappa, y_true, y_pred = evaluate_personalized(
        models,
        clients,
        domain_model=False,
    )

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for cid, model in models.items():
        torch.save(model.state_dict(), seed_dir / f"pers_fl_client_{cid}.pt")

    save_classification_report(y_true, y_pred, seed_dir / "pers_fl_report.txt")

    return {
        "method": "Pers. FL (no DA)",
        "seed": seed,
        "accuracy": acc,
        "macro_f1": f1,
        "kappa": kappa,
    }


# =============================================================================
# DA-PFL
# =============================================================================

def train_local_da(global_model, train_loader, client_id: int):
    """
    Phase 1 DA-PFL local training.

    The scalar loss is L_cls + L_dom. Since L_dom flows through the GRL,
    the feature extractor receives the effective adversarial gradient.
    """
    local_model = copy.deepcopy(global_model).to(DEVICE)

    optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
    cls_loss_fn = nn.CrossEntropyLoss()
    dom_loss_fn = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        local_model.train()

        for X_b, y_b in tqdm(train_loader, desc=f"DA local cid={client_id}", leave=False):
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            domain_labels = torch.full(
                (X_b.size(0),),
                int(client_id),
                dtype=torch.long,
                device=DEVICE,
            )

            optimizer.zero_grad()

            class_logits, domain_logits = local_model(X_b)
            loss = cls_loss_fn(class_logits, y_b) + dom_loss_fn(domain_logits, domain_labels)

            loss.backward()
            optimizer.step()

    return local_model.state_dict()


def run_da_pfl_phase1(clients, seed: int, num_domains: int, out_dir: Path, verbose: bool = False):
    set_seed(seed)

    global_model = DomainAdaptiveCNNBiLSTM(
        num_classes=5,
        num_domains=num_domains,
        lam=DOMAIN_LAMBDA,
    ).to(DEVICE)

    convergence = []

    for rnd in range(FED_ROUNDS):
        local_states, client_sizes = [], []

        for cid, data in clients.items():
            loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
            local_states.append(train_local_da(global_model, loader, cid))
            client_sizes.append(data["num_train"])

        global_model.load_state_dict(fedavg(local_states, client_sizes))

        acc, f1, kappa, _, _ = aggregate_global_eval(
            global_model,
            clients,
            domain_model=True,
        )

        convergence.append({
            "method": "DA-PFL pre-personalization",
            "seed": seed,
            "round": rnd + 1,
            "accuracy": acc,
            "macro_f1": f1,
            "kappa": kappa,
        })

        if verbose:
            print(
                f"DA-PFL Phase 1 | seed={seed} | round={rnd+1}/{FED_ROUNDS} "
                f"| acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f}"
            )

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), seed_dir / "da_pfl_phase1_grl.pt")

    return global_model, convergence


def personalize_da_pfl(global_model, clients):
    """
    Phase 2 DA-PFL personalization.

    Important paper fix:
    - No domain loss here.
    - Domain logits are ignored.
    - Fine-tuning uses classification loss only.
    """
    personalized = {}

    for cid, data in clients.items():
        model = copy.deepcopy(global_model).to(DEVICE)
        loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR * PERSONAL_LR_FACTOR)
        criterion = nn.CrossEntropyLoss()

        for ep in range(PERSONAL_EPOCHS):
            model.train()

            for X_b, y_b in tqdm(loader, desc=f"DA-PFL Phase2 cid={cid} ep={ep+1}", leave=False):
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

                optimizer.zero_grad()
                class_logits, _ = model(X_b)
                loss = criterion(class_logits, y_b)
                loss.backward()
                optimizer.step()

        personalized[cid] = model

    return personalized


def run_da_pfl(global_model, clients, seed: int, out_dir: Path):
    models = personalize_da_pfl(global_model, clients)

    acc, f1, kappa, y_true, y_pred = evaluate_personalized(
        models,
        clients,
        domain_model=True,
    )

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for cid, model in models.items():
        torch.save(model.state_dict(), seed_dir / f"da_pfl_client_{cid}.pt")

    save_classification_report(y_true, y_pred, seed_dir / "da_pfl_report.txt")

    return {
        "method": "DA-PFL (ours)",
        "seed": seed,
        "accuracy": acc,
        "macro_f1": f1,
        "kappa": kappa,
    }


# =============================================================================
# Result summarization
# =============================================================================

def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, group in results_df.groupby("method", sort=False):
        rows.append({
            "method": method,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=1) if len(group) > 1 else 0.0,
            "macro_f1_mean": group["macro_f1"].mean(),
            "macro_f1_std": group["macro_f1"].std(ddof=1) if len(group) > 1 else 0.0,
            "kappa_mean": group["kappa"].mean(),
            "kappa_std": group["kappa"].std(ddof=1) if len(group) > 1 else 0.0,
            "n_seeds": len(group),
        })

    return pd.DataFrame(rows)


def format_latex_table(summary_df: pd.DataFrame) -> str:
    lines = []

    for _, row in summary_df.iterrows():
        line = (
            f"{row['method']} & "
            f"${row['accuracy_mean']:.4f} \\pm {row['accuracy_std']:.4f}$ & "
            f"${row['macro_f1_mean']:.4f} \\pm {row['macro_f1_std']:.4f}$ & "
            f"${row['kappa_mean']:.4f} \\pm {row['kappa_std']:.4f}$ \\\\"
        )
        lines.append(line)

    return "\n".join(lines)


def compute_delta_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    indexed = summary_df.set_index("method")
    rows = []

    def add_delta(name, method_a, method_b):
        if method_a not in indexed.index or method_b not in indexed.index:
            return
        a = indexed.loc[method_a]
        b = indexed.loc[method_b]
        rows.append({
            "comparison": name,
            "delta_accuracy": a["accuracy_mean"] - b["accuracy_mean"],
            "delta_macro_f1": a["macro_f1_mean"] - b["macro_f1_mean"],
            "delta_kappa": a["kappa_mean"] - b["kappa_mean"],
        })

    add_delta("DA-PFL vs. FedAvg", "DA-PFL (ours)", "FedAvg")
    add_delta("DA-PFL vs. Pers. FL", "DA-PFL (ours)", "Pers. FL (no DA)")
    add_delta("DA-PFL vs. Centralized", "DA-PFL (ours)", "Centralized CNN-BiLSTM")

    return pd.DataFrame(rows)


def save_experiment_metadata(args, X, y, client_ids, out_dir: Path):
    labels, counts = np.unique(y, return_counts=True)

    metadata = {
        "device": DEVICE,
        "subjects": SUBJECTS,
        "records": RECORDS,
        "channel": CHANNEL,
        "epoch_sec": EPOCH_SEC,
        "sfreq": SFREQ,
        "batch_size": BATCH_SIZE,
        "central_epochs": CENTRAL_EPOCHS,
        "fed_rounds": FED_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "personal_epochs": PERSONAL_EPOCHS,
        "lr": LR,
        "personal_lr_factor": PERSONAL_LR_FACTOR,
        "domain_lambda": DOMAIN_LAMBDA,
        "seeds": args.seeds,
        "num_samples": int(len(y)),
        "num_clients": int(len(np.unique(client_ids))),
        "class_distribution": {
            ID_2_LABEL[int(label)]: int(count)
            for label, count in zip(labels, counts)
        },
    }

    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))


# =============================================================================
# Main experiment loop
# =============================================================================

def run_all_experiments(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading Sleep-EDF data. This may download the dataset the first time.")
    X, y, client_ids = load_sleep_edf()
    num_domains = len(np.unique(client_ids))

    print(f"Data shape: {X.shape}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Number of clients: {num_domains}")

    save_experiment_metadata(args, X, y, client_ids, out_dir)

    all_results = []
    all_convergence = []

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"Running seed {seed}")
        print("=" * 80)

        set_seed(seed)
        clients = split_clients(X, y, client_ids, seed=seed)

        if not args.skip_centralized:
            print("\n[1/4] Centralized CNN-BiLSTM")
            central_result = run_centralized(
                X,
                y,
                seed=seed,
                out_dir=out_dir,
                verbose=args.verbose,
            )
            all_results.append(central_result)

        print("\n[2/4] FedAvg")
        fedavg_model, fedavg_result, fedavg_curve = run_fedavg(
            clients,
            seed=seed,
            out_dir=out_dir,
            verbose=args.verbose,
        )
        all_results.append(fedavg_result)
        all_convergence.extend(fedavg_curve)

        print("\n[3/4] Personalized FL")
        pers_result = run_personalized_fl(
            fedavg_model,
            clients,
            seed=seed,
            out_dir=out_dir,
        )
        all_results.append(pers_result)

        print("\n[4/4] DA-PFL")
        da_phase1_model, da_curve = run_da_pfl_phase1(
            clients,
            seed=seed,
            num_domains=num_domains,
            out_dir=out_dir,
            verbose=args.verbose,
        )
        all_convergence.extend(da_curve)

        da_result = run_da_pfl(
            da_phase1_model,
            clients,
            seed=seed,
            out_dir=out_dir,
        )
        all_results.append(da_result)

        seed_df = pd.DataFrame([r for r in all_results if r["seed"] == seed])
        print("\nSeed summary:")
        print(seed_df[["method", "accuracy", "macro_f1", "kappa"]].to_string(index=False))

    results_df = pd.DataFrame(all_results)
    convergence_df = pd.DataFrame(all_convergence)

    results_df.to_csv(out_dir / "per_seed_results.csv", index=False)
    convergence_df.to_csv(out_dir / "convergence_curves.csv", index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(out_dir / "summary_mean_std.csv", index=False)

    delta_df = compute_delta_rows(summary_df)
    delta_df.to_csv(out_dir / "delta_rows.csv", index=False)

    latex_rows = format_latex_table(summary_df)
    (out_dir / "latex_table_rows.txt").write_text(latex_rows)

    print("\n" + "=" * 80)
    print("FINAL MULTI-SEED SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("DELTA ROWS")
    print("=" * 80)
    print(delta_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("LATEX TABLE ROWS")
    print("=" * 80)
    print(latex_rows)

    print(f"\nSaved results to: {out_dir.resolve()}")

    return results_df, summary_df, convergence_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for multi-seed evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="paper_results",
        help="Directory to save results and checkpoints.",
    )
    parser.add_argument(
        "--skip-centralized",
        action="store_true",
        help="Skip centralized baseline to save runtime.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print epoch/round metrics.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)
