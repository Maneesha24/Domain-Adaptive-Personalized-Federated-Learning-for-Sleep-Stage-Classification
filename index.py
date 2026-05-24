"""
Personalized Federated Learning for Sleep Stage Classification.

This module implements and compares four learning paradigms on the Sleep-EDF
(PhysioNet) dataset:

    1. **Centralized CNN-BiLSTM** — upper-bound baseline with pooled data.
    2. **FedAvg** — standard federated averaging across per-subject clients.
    3. **Personalized FL** — FedAvg global model + per-client fine-tuning.
    4. **DA-PFL** — domain-adaptive federated learning with a Gradient
       Reversal Layer (GRL) for domain-invariant features, followed by
       client personalization.

Each client corresponds to one subject's recordings. EEG epochs are extracted
from a single channel (Fpz-Cz), band-pass filtered, resampled, and z-scored
before being fed into a shared CNN-BiLSTM backbone.

References
----------
    Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by
    backpropagation. *ICML*.

Usage
-----
    python index.py

Trained model checkpoints are written to the ``models/`` directory.
"""

import os
import copy
import warnings
import numpy as np
import mne
import torch
import torch.nn as nn

from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, classification_report
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================
# Global hyperparameters and label mappings for Sleep-EDF preprocessing
# and all training/evaluation pipelines.

# --- Data / preprocessing ---
SUBJECTS = list(range(10))   # PhysioNet subject IDs 0–9
RECORDS  = [1, 2]            # Two PSG recordings per subject
CHANNEL  = "EEG Fpz-Cz"      # Single-channel montage used for classification
EPOCH_SEC = 30               # Hypnogram epoch length (seconds)
SFREQ     = 100              # Target sampling rate after resampling (Hz)

# --- Training schedule ---
BATCH_SIZE      = 64         # Mini-batch size for all DataLoaders
CENTRAL_EPOCHS  = 10         # Epochs for centralized upper-bound baseline
FED_ROUNDS      = 10         # Federated communication rounds (FedAvg / DA-PFL)
LOCAL_EPOCHS    = 1          # Local epochs per client per federated round
PERSONAL_EPOCHS = 3          # Client fine-tuning epochs after federation

# --- Optimisation ---
LR            = 1e-3         # Adam learning rate (local & centralized)
DOMAIN_LAMBDA = 0.1          # GRL reversal scale λ (DA-PFL adversarial strength)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Maps PhysioNet annotation strings → integer class IDs (N3/N4 merged).
ANNOTATION_DESC_2_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # merged into N3
    "Sleep stage R": 4,
}
# Human-readable stage names for classification reports.
ID_2_LABEL = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# =============================================================================
# Gradient Reversal Layer  (Ganin & Lempitsky, ICML 2015)
# =============================================================================
class _GradientReversalFunction(Function):
    """
    Custom autograd function implementing the Gradient Reversal Layer (GRL).

    During the forward pass the input is passed through unchanged (identity).
    During the backward pass gradients w.r.t. the input are negated and scaled
    by ``lam``, so upstream layers effectively *maximize* domain loss while
    downstream layers *minimize* it.

    @see GradientReversalLayer
    @see DomainAdaptiveCNNBiLSTM
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Identity forward: output equals input.

        @param ctx: Autograd context; stores ``lam`` for backward.
        @param x: Feature tensor of shape ``(B, D)``.
        @param lam: Reversal scale λ (scalar); controls adversarial strength.
        @return: Clone of ``x`` (same shape as input).
        """
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Reverse and scale gradients flowing into shared features.

        @param ctx: Autograd context with saved ``lam``.
        @param grad_output: Upstream gradient from domain head, shape matches ``x``.
        @return: Tuple ``(grad_input, None)`` where ``grad_input = -λ * grad_output``
                 and the second element is ``None`` (``lam`` is not differentiable).
        """
        (lam,) = ctx.saved_tensors
        return -lam.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    ``nn.Module`` wrapper around :class:`_GradientReversalFunction`.

    Inserts adversarial domain adaptation between a shared feature extractor
    and a domain classifier. Feature layers upstream of this module receive
    reversed domain gradients; the domain head receives normal gradients.

    @param lam: Reversal scale λ (default 1.0). Larger values strengthen
                domain-invariance pressure on the backbone.
    """

    def __init__(self, lam: float = 1.0):
        """
        @param lam: Gradient reversal multiplier (λ in Ganin & Lempitsky, 2015).
        """
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient reversal on the backward pass only.

        @param x: Shared features ``h`` of shape ``(batch, 128)``.
        @return: Same tensor as ``x`` in the forward pass.
        """
        return _GradientReversalFunction.apply(x, self.lam)

    def extra_repr(self) -> str:
        """@return: Human-readable representation of λ."""
        return f"lambda={self.lam}"


# =============================================================================
# Dataset Loading
# =============================================================================
def load_sleep_edf(subjects=SUBJECTS, records=RECORDS):
    """
    Download, preprocess, and concatenate Sleep-EDF recordings into arrays.

    For each subject/recording pair the pipeline: fetches EDF + hypnogram,
    annotates sleep stages, selects ``CHANNEL``, filters 0.3–35 Hz, resamples
    to ``SFREQ`` Hz, segments into ``EPOCH_SEC``-second epochs, and applies
    per-epoch z-score normalization along the time axis.

    @param subjects: List of PhysioNet subject indices (default ``SUBJECTS``).
    @param records: List of recording indices per subject (default ``RECORDS``).
    @return: Tuple ``(X, y, client_ids)`` where
             - ``X``: float32 array ``(N, 1, T)`` — EEG epochs (1 channel).
             - ``y``: int64 array ``(N,)`` — stage labels in ``[0, 4]``.
             - ``client_ids``: int array ``(N,)`` — federated client index per epoch.
    @raises ImportError: If ``mne`` sleep_physionet dataset helpers are unavailable.
    @note Stages 3 and 4 are merged into class 3 (N3) via ``ANNOTATION_DESC_2_ID``.
    """
    from mne.datasets.sleep_physionet.age import fetch_data

    X_all, y_all, client_ids = [], [], []
    files = fetch_data(subjects=subjects, recording=records)

    for client_idx, file_pair in enumerate(files):
        psg_file, hyp_file = file_pair

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
            raw, events, event_id=event_id,
            tmin=0, tmax=EPOCH_SEC - 1.0 / SFREQ,
            baseline=None, preload=True, verbose=False,
        )

        X = epochs.get_data()
        y = epochs.events[:, -1]

        # per-epoch z-score normalisation
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


# =============================================================================
# Dataset Class
# =============================================================================
class SleepDataset(Dataset):
    """
    PyTorch ``Dataset`` for preprocessed Sleep-EDF epoch tensors.

    Wraps in-memory NumPy arrays as ``float32`` features and ``long`` labels
    suitable for ``DataLoader`` batching.

    @param X: EEG epochs, shape ``(N, 1, time_steps)`` or ``(N, C, T)``.
    @param y: Integer sleep-stage labels, shape ``(N,)``.
    """

    def __init__(self, X, y):
        """
        @param X: Feature array (converted to ``torch.float32``).
        @param y: Label array (converted to ``torch.long``).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """@return: Number of epochs in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        @param idx: Sample index in ``[0, len(self))``.
        @return: Tuple ``(X[idx], y[idx])`` of tensors.
        """
        return self.X[idx], self.y[idx]


# =============================================================================
# Base CNN-BiLSTM  (used by Centralized, FedAvg, Personalized FL)
# =============================================================================
class CNNBiLSTM(nn.Module):
    """
    Convolutional + bidirectional LSTM classifier for single-channel EEG epochs.

    Architecture:
        Conv1d stack (32 → 64 → 128 channels, max-pool) → BiLSTM (hidden 64) →
        last time-step features (128-d) → MLP classifier.

    Used as the backbone for centralized training, FedAvg, personalized FL,
    and as the feature extractor inside :class:`DomainAdaptiveCNNBiLSTM`.

    @param num_classes: Number of sleep stages (default 5: W, N1, N2, N3, REM).
    """

    def __init__(self, num_classes: int = 5):
        """
        @param num_classes: Output dimension of the classification head.
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run CNN + BiLSTM and return the final hidden state (shared embedding).

        @param x: Input batch ``(B, 1, T)`` — one EEG channel per epoch.
        @return: Feature tensor ``(B, 128)`` (concatenated BiLSTM directions).
        """
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: features → linear classifier.

        @param x: Input batch ``(B, 1, T)``.
        @return: Class logits ``(B, num_classes)``.
        """
        return self.classifier(self.extract_features(x))


# =============================================================================
# Domain-Adaptive CNN-BiLSTM  — with proper Gradient Reversal Layer
# =============================================================================
class DomainAdaptiveCNNBiLSTM(nn.Module):
    """
    Domain-adaptive CNN-BiLSTM with dual heads and Gradient Reversal Layer.

    Architecture (matches Section III-B of the paper):

        EEG epoch → CNN → BiLSTM → h (128-d shared feature)
                                    │
                         ┌──────────┴──────────────┐
                         │                         │
                  Classifier head          GRL(λ) → Domain head
                  (sleep stage)            (client identity)

    The GRL reverses gradients flowing back from the domain head into ``h``,
    so the shared feature extractor is simultaneously optimised to:

        - **Minimise** ``L_cls`` (correct stage prediction)
        - **Maximise** ``L_dom`` (confuse the domain discriminator)

    Effective feature loss (eq. 3):

        ``L_{f_θ} = L_cls(g_φ(h), y) − λ · L_dom(d_ψ(h), k)``

    Implemented in one forward/backward pass via :class:`GradientReversalLayer`.

    @param num_classes: Sleep stage classes (default 5).
    @param num_domains: Number of federated clients / domain labels.
    @param lam: GRL reversal scale λ (default ``DOMAIN_LAMBDA``).
    @see train_local_da_model
    """

    def __init__(self, num_classes: int = 5, num_domains: int = 20,
                 lam: float = DOMAIN_LAMBDA):
        """
        @param num_classes: Output size of the sleep-stage classifier.
        @param num_domains: Output size of the domain (client-id) classifier.
        @param lam: Adversarial strength passed to :class:`GradientReversalLayer`.
        """
        super().__init__()

        self.backbone = CNNBiLSTM(num_classes=num_classes)

        # GRL sits between the shared feature and the domain head
        self.grl = GradientReversalLayer(lam=lam)

        # Domain discriminator  h → GRL(h) → d_ψ  (predicts client id)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor):
        """
        Dual-head forward: classification on ``h``, domain prediction on ``GRL(h)``.

        @param x: EEG batch ``(B, 1, T)``.
        @return: Tuple ``(class_logits, domain_logits)`` where
                 - ``class_logits``: ``(B, num_classes)`` sleep stage scores.
                 - ``domain_logits``: ``(B, num_domains)`` client-id scores
                   (gradients into ``h`` are reversed via GRL).
        """
        h = self.backbone.extract_features(x)
        class_logits  = self.backbone.classifier(h)
        domain_logits = self.domain_classifier(self.grl(h))
        return class_logits, domain_logits


# =============================================================================
# Training / Evaluation Helpers
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train a standard (non-domain-adaptive) model for one epoch.

    @param model: ``nn.Module`` with single-output ``forward`` (e.g. :class:`CNNBiLSTM`).
    @param loader: ``DataLoader`` yielding ``(X, y)`` batches.
    @param optimizer: PyTorch optimizer (e.g. Adam).
    @param criterion: Loss function (typically ``CrossEntropyLoss``).
    @return: Mean training loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    """
    Evaluate a single-output classifier on a held-out loader.

    @param model: Classifier producing logits ``(B, num_classes)``.
    @param loader: Test ``DataLoader`` with ``(X, y)`` batches.
    @return: Tuple ``(accuracy, macro_f1, kappa, y_true, y_pred)`` where the
             last two are Python lists of integer labels for reporting.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = torch.argmax(model(X.to(DEVICE)), dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        cohen_kappa_score(y_true, y_pred),
        y_true, y_pred,
    )


def evaluate_domain_model(model, loader):
    """
    Evaluate :class:`DomainAdaptiveCNNBiLSTM` using only the classification head.

    Domain logits are computed but discarded; metrics reflect sleep-stage accuracy.

    @param model: :class:`DomainAdaptiveCNNBiLSTM` instance.
    @param loader: Test ``DataLoader`` with ``(X, y)`` batches.
    @return: Tuple ``(accuracy, macro_f1, kappa, y_true, y_pred)``.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            class_logits, _ = model(X.to(DEVICE))
            preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        cohen_kappa_score(y_true, y_pred),
        y_true, y_pred,
    )


# =============================================================================
# Client Split
# =============================================================================
def split_clients(X, y, client_ids, test_size=0.2):
    """
    Partition pooled data into per-client train/test :class:`SleepDataset` splits.

    Each unique value in ``client_ids`` becomes one federated client. Stratified
    ``train_test_split`` preserves label distribution within each client.

    @param X: Full feature array ``(N, 1, T)``.
    @param y: Full label array ``(N,)``.
    @param client_ids: Client index per sample ``(N,)``.
    @param test_size: Fraction of each client's epochs reserved for testing.
    @return: Dict mapping ``client_id → {"train", "test", "num_train"}``.
    """
    clients = {}
    for cid in np.unique(client_ids):
        idx = np.where(client_ids == cid)[0]
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=42, stratify=y[idx]
        )
        clients[int(cid)] = {
            "train": SleepDataset(X[train_idx], y[train_idx]),
            "test":  SleepDataset(X[test_idx],  y[test_idx]),
            "num_train": len(train_idx),
        }
    return clients


# =============================================================================
# FedAvg Aggregation
# =============================================================================
def fedavg(state_dicts, client_sizes):
    """
    Federated averaging (FedAvg) over client model weights.

    Each parameter is a sample-size-weighted mean across clients:

        ``θ_global = Σ_i (n_i / N) · θ_i``

    @param state_dicts: List of per-client ``state_dict`` mappings.
    @param client_sizes: List of training sample counts ``n_i`` per client.
    @return: Aggregated ``state_dict`` with the same keys as client dicts.
    """
    avg_state = copy.deepcopy(state_dicts[0])
    total = sum(client_sizes)
    for key in avg_state:
        avg_state[key] = sum(
            state_dicts[i][key] * (client_sizes[i] / total)
            for i in range(len(state_dicts))
        )
    return avg_state


def train_local_model(global_model, train_loader, epochs=1):
    """
    One round of local FedAvg client training (classification only).

    Copies the global model, runs ``epochs`` of SGD on the client's private
    data, and returns updated weights without modifying the global model in-place.

    @param global_model: Server/global :class:`CNNBiLSTM` (or compatible module).
    @param train_loader: Client training ``DataLoader``.
    @param epochs: Number of local passes over the client data (default 1).
    @return: ``state_dict`` of the locally trained model.
    """
    local = copy.deepcopy(global_model).to(DEVICE)
    opt   = torch.optim.Adam(local.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_one_epoch(local, train_loader, opt, crit)
    return local.state_dict()


def evaluate_global_model(model, clients):
    """
    Pool predictions from all clients' test sets for one global FedAvg model.

    @param model: Global :class:`CNNBiLSTM` after aggregation.
    @param clients: Client dict from :func:`split_clients`.
    @return: Tuple ``(accuracy, macro_f1, kappa)`` over concatenated test epochs.
    """
    all_true, all_pred = [], []
    for data in clients.values():
        loader = DataLoader(data["test"], batch_size=BATCH_SIZE, shuffle=False)
        _, _, _, yt, yp = evaluate(model, loader)
        all_true.extend(yt); all_pred.extend(yp)
    return (
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro"),
        cohen_kappa_score(all_true, all_pred),
    )


def run_fedavg(clients):
    """
    Run full FedAvg training for ``FED_ROUNDS`` communication rounds.

    Each round: every client trains locally on its private data, uploads
    weights, server aggregates via :func:`fedavg`, then evaluates globally.

    @param clients: Per-client train/test data from :func:`split_clients`.
    @return: Trained global :class:`CNNBiLSTM` after the final aggregation round.
    """
    global_model = CNNBiLSTM(num_classes=5).to(DEVICE)
    print("\n========== FedAvg Training ==========")
    for rnd in range(FED_ROUNDS):
        states, sizes = [], []
        for data in clients.values():
            loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
            states.append(train_local_model(global_model, loader, LOCAL_EPOCHS))
            sizes.append(data["num_train"])
        global_model.load_state_dict(fedavg(states, sizes))
        acc, f1, kappa = evaluate_global_model(global_model, clients)
        print(f"FedAvg Round {rnd+1}/{FED_ROUNDS} | "
              f"Acc: {acc:.4f} | Macro-F1: {f1:.4f} | Kappa: {kappa:.4f}")
    return global_model


# =============================================================================
# Personalized FL  (FedAvg global + client fine-tune)
# =============================================================================
def personalize_clients(global_model, clients, fine_tune_epochs=3):
    """
    Personalized FL: fine-tune a copy of the global model per client.

    Uses a reduced learning rate (``LR * 0.1``) and client-local training data
    only. No domain-adaptation loss is applied.

    @param global_model: FedAvg-global :class:`CNNBiLSTM`.
    @param clients: Client dict from :func:`split_clients`.
    @param fine_tune_epochs: Local fine-tuning epochs per client (default 3).
    @return: Dict ``client_id → personalized nn.Module``.
    """
    personalized = {}
    for cid, data in clients.items():
        model = copy.deepcopy(global_model).to(DEVICE)
        loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
        opt  = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
        crit = nn.CrossEntropyLoss()
        for _ in range(fine_tune_epochs):
            train_one_epoch(model, loader, opt, crit)
        personalized[cid] = model
    return personalized


def evaluate_personalized_models(models, clients):
    """
    Evaluate per-client personalized models on each client's test split.

    @param models: Dict from :func:`personalize_clients`.
    @param clients: Client dict from :func:`split_clients`.
    @return: Tuple ``(accuracy, macro_f1, kappa)`` pooled over all clients.
    """
    all_true, all_pred = [], []
    for cid, model in models.items():
        loader = DataLoader(clients[cid]["test"], batch_size=BATCH_SIZE, shuffle=False)
        _, _, _, yt, yp = evaluate(model, loader)
        all_true.extend(yt); all_pred.extend(yp)
    return (
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro"),
        cohen_kappa_score(all_true, all_pred),
    )


# =============================================================================
# DA-PFL  — Domain-Adaptive Federated Learning with Gradient Reversal Layer
# =============================================================================
def train_local_da_model(global_model, train_loader, client_id, epochs=1):
    """
    Local DA-PFL training round with joint classification + domain losses.

    Combined loss (scalar sum for one backward pass):

        ``L = L_cls(g_φ(h), y) + L_dom(d_ψ(GRL_λ(h)), k)``

    GRL negates gradients into ``h``, so the effective feature update is:

        ``∂L/∂θ = ∂L_cls/∂θ − λ · ∂L_dom/∂θ``

    The backbone minimises stage loss and maximises domain confusion; the
    domain head still minimises ``L_dom`` with normal gradients.

    @param global_model: Global :class:`DomainAdaptiveCNNBiLSTM`.
    @param train_loader: Client training ``DataLoader``.
    @param client_id: Integer client/domain label ``k`` for all batches in this round.
    @param epochs: Local training epochs (default 1).
    @return: ``state_dict`` after local adversarial + classification training.
    @see DomainAdaptiveCNNBiLSTM
    """
    local = copy.deepcopy(global_model).to(DEVICE)
    opt   = torch.optim.Adam(local.parameters(), lr=LR)

    cls_loss_fn = nn.CrossEntropyLoss()
    dom_loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        local.train()
        for X_b, y_b in tqdm(train_loader, desc=f"DA-Local cid={client_id}", leave=False):
            X_b = X_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            domain_labels = torch.full(
                (X_b.size(0),), int(client_id),
                dtype=torch.long, device=DEVICE,
            )

            opt.zero_grad()

            # Forward: class_logits uses h directly;
            #          domain_logits uses GRL(h) — see DomainAdaptiveCNNBiLSTM.forward()
            class_logits, domain_logits = local(X_b)

            # L_cls  →  normal minimisation for classifier + feature extractor
            # L_dom  →  normal minimisation for domain head;
            #           gradient REVERSAL via GRL causes feature extractor
            #           to MAXIMISE this term (domain confusion)
            loss = cls_loss_fn(class_logits, y_b) + dom_loss_fn(domain_logits, domain_labels)

            loss.backward()   # GRL handles sign reversal automatically
            opt.step()

    return local.state_dict()


def evaluate_da_global_model(model, clients):
    """
    Evaluate a global DA-PFL model across all clients' test sets.

    @param model: Global :class:`DomainAdaptiveCNNBiLSTM`.
    @param clients: Client dict from :func:`split_clients`.
    @return: Tuple ``(accuracy, macro_f1, kappa)`` using classification head only.
    """
    all_true, all_pred = [], []
    for data in clients.values():
        loader = DataLoader(data["test"], batch_size=BATCH_SIZE, shuffle=False)
        _, _, _, yt, yp = evaluate_domain_model(model, loader)
        all_true.extend(yt); all_pred.extend(yp)
    return (
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro"),
        cohen_kappa_score(all_true, all_pred),
    )


def run_da_pfl(clients, num_domains):
    """
    Run domain-adaptive federated learning (DA-PFL) for ``FED_ROUNDS`` rounds.

    Same communication pattern as :func:`run_fedavg`, but local updates use
    :func:`train_local_da_model` with GRL-based domain adaptation.

    @param clients: Per-client data from :func:`split_clients`.
    @param num_domains: Number of clients (domain classifier output size).
    @return: Trained global :class:`DomainAdaptiveCNNBiLSTM`.
    """
    global_model = DomainAdaptiveCNNBiLSTM(
        num_classes=5, num_domains=num_domains, lam=DOMAIN_LAMBDA
    ).to(DEVICE)

    print("\n========== DA-PFL: Federated Training ==========")
    for rnd in range(FED_ROUNDS):
        states, sizes = [], []
        for cid, data in clients.items():
            loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
            states.append(train_local_da_model(global_model, loader, cid, LOCAL_EPOCHS))
            sizes.append(data["num_train"])
        global_model.load_state_dict(fedavg(states, sizes))
        acc, f1, kappa = evaluate_da_global_model(global_model, clients)
        print(f"DA-PFL Round {rnd+1}/{FED_ROUNDS} | "
              f"Acc: {acc:.4f} | Macro-F1: {f1:.4f} | Kappa: {kappa:.4f}")
    return global_model


def personalize_da_clients(global_model, clients, fine_tune_epochs=3):
    """
    Phase 2 of DA-PFL: per-client fine-tuning of the global DA model.

    Continues joint ``L_cls + L_dom`` training with domain labels set to each
    client's id, using learning rate ``LR * 0.1``.

    @param global_model: Global DA-PFL model after :func:`run_da_pfl`.
    @param clients: Client dict from :func:`split_clients`.
    @param fine_tune_epochs: Personalization epochs per client (default 3).
    @return: Dict ``client_id → personalized :class:`DomainAdaptiveCNNBiLSTM``.
    """
    personalized = {}
    for cid, data in clients.items():
        model = copy.deepcopy(global_model).to(DEVICE)
        loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
        opt         = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
        cls_loss_fn = nn.CrossEntropyLoss()
        dom_loss_fn = nn.CrossEntropyLoss()

        for _ in range(fine_tune_epochs):
            model.train()
            for X_b, y_b in tqdm(loader, desc=f"Personalize cid={cid}", leave=False):
                X_b = X_b.to(DEVICE)
                y_b = y_b.to(DEVICE)
                domain_labels = torch.full(
                    (X_b.size(0),), int(cid),
                    dtype=torch.long, device=DEVICE,
                )
                opt.zero_grad()
                class_logits, domain_logits = model(X_b)
                loss = cls_loss_fn(class_logits, y_b) + dom_loss_fn(domain_logits, domain_labels)
                loss.backward()
                opt.step()

        personalized[cid] = model
    return personalized


def evaluate_personalized_da_models(models, clients):
    """
    Evaluate personalized DA-PFL models on each client's held-out data.

    @param models: Dict from :func:`personalize_da_clients`.
    @param clients: Client dict from :func:`split_clients`.
    @return: Tuple ``(accuracy, macro_f1, kappa)`` pooled over all test sets.
    """
    all_true, all_pred = [], []
    for cid, model in models.items():
        loader = DataLoader(clients[cid]["test"], batch_size=BATCH_SIZE, shuffle=False)
        _, _, _, yt, yp = evaluate_domain_model(model, loader)
        all_true.extend(yt); all_pred.extend(yp)
    return (
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro"),
        cohen_kappa_score(all_true, all_pred),
    )


# =============================================================================
# Centralized Baseline
# =============================================================================
def run_centralized_baseline(X, y):
    """
    Train and evaluate a centralized CNN-BiLSTM upper-bound baseline.

    Performs a single stratified 80/20 train/test split on the full pooled
    dataset (ignoring client boundaries), trains for ``CENTRAL_EPOCHS``, and
    prints a sklearn ``classification_report``.

    @param X: Full feature array from :func:`load_sleep_edf`.
    @param y: Full label array.
    @return: Tuple ``(model, accuracy, macro_f1, kappa)`` for the best epoch.
    """
    print("\n========== Centralized CNN-BiLSTM ==========")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    train_loader = DataLoader(SleepDataset(X[train_idx], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SleepDataset(X[test_idx],  y[test_idx]),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = CNNBiLSTM(num_classes=5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(CENTRAL_EPOCHS):
        loss = train_one_epoch(model, train_loader, opt, crit)
        acc, f1, kappa, _, _ = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{CENTRAL_EPOCHS} | Loss: {loss:.4f} | "
              f"Acc: {acc:.4f} | Macro-F1: {f1:.4f} | Kappa: {kappa:.4f}")

    acc, f1, kappa, yt, yp = evaluate(model, test_loader)
    print("\nCentralized Final Results")
    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print("Cohen Kappa:", kappa)
    print(classification_report(yt, yp, target_names=[ID_2_LABEL[i] for i in range(5)]))
    return model, acc, f1, kappa


# =============================================================================
# Main
# =============================================================================
def main():
    """
    End-to-end experiment driver: load data, train all four paradigms, compare.

    Execution order:
        1. Centralized baseline → save ``models/centralized_cnn_bilstm.pt``
        2. FedAvg → save ``models/fedavg_sleep_edf.pt``
        3. Personalized FL (FedAvg + fine-tune)
        4. DA-PFL (GRL federation + personalization) → save ``models/da_pfl_grl.pt``

    Prints per-round metrics during training and a final comparison table
    (accuracy, macro-F1, Cohen's kappa) for all methods.

    @return: None (results printed to stdout; checkpoints on disk).
    """
    os.makedirs("models", exist_ok=True)

    print("Downloading and preprocessing Sleep-EDF...")
    X, y, client_ids = load_sleep_edf()
    print("Data shape:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))
    print("Number of clients:", len(np.unique(client_ids)))

    results = {}

    # 1. Centralized upper bound
    centralized_model, acc, f1, kappa = run_centralized_baseline(X, y)
    results["Centralized CNN-BiLSTM"] = (acc, f1, kappa)
    torch.save(centralized_model.state_dict(), "models/centralized_cnn_bilstm.pt")

    clients = split_clients(X, y, client_ids)

    # 2. Standard FedAvg
    fedavg_model = run_fedavg(clients)
    acc, f1, kappa = evaluate_global_model(fedavg_model, clients)
    results["FedAvg"] = (acc, f1, kappa)
    torch.save(fedavg_model.state_dict(), "models/fedavg_sleep_edf.pt")

    # 3. Personalized FL (FedAvg global + client fine-tune, no domain adapt.)
    pers_models = personalize_clients(fedavg_model, clients, PERSONAL_EPOCHS)
    acc, f1, kappa = evaluate_personalized_models(pers_models, clients)
    results["Personalized FL (no DA)"] = (acc, f1, kappa)

    # 4. DA-PFL: GRL-based adversarial domain adaptation + personalization
    num_domains = len(np.unique(client_ids))
    da_model = run_da_pfl(clients, num_domains)

    pers_da_models = personalize_da_clients(da_model, clients, PERSONAL_EPOCHS)
    acc, f1, kappa = evaluate_personalized_da_models(pers_da_models, clients)
    results["DA-PFL (GRL + Personalization)"] = (acc, f1, kappa)
    torch.save(da_model.state_dict(), "models/da_pfl_grl.pt")

    # Summary
    print("\n========== Final Comparison ==========")
    print(f"{'Model':<45} | {'Accuracy':>10} | {'Macro-F1':>10} | {'Kappa':>10}")
    print("-" * 90)
    for name, (acc, f1, kappa) in results.items():
        print(f"{name:<45} | {acc:.4f}     | {f1:.4f}     | {kappa:.4f}")


if __name__ == "__main__":
    main()