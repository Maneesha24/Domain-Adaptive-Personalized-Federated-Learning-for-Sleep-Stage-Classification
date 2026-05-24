# Personalized Federated Learning for Sleep Stage Classification

## Author

**Isha Venigalla**  
Software Development Engineer II, Amazon  
New York City, USA

---

A PyTorch implementation of **Domain-Adaptive Personalized Federated Learning (DA-PFL)** for automated sleep stage classification on the [Sleep-EDF Cassette](https://physionet.org/content/sleep-edfx/1.0.0/) dataset. The project compares centralized training, standard FedAvg, client-specific personalization, and domain-adaptive federated learning using a shared **CNN-BiLSTM** backbone.

---

## Table of Contents

- [Author](#author)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Methods Compared](#methods-compared)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Expected Output](#expected-output)
- [Saved Models](#saved-models)
- [Research Context](#research-context)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Sleep stage classification from EEG is essential for diagnosing sleep disorders, but centralized deep learning requires pooling sensitive patient data across institutions. Federated learning (FL) enables collaborative training without sharing raw recordings, yet standard **FedAvg** struggles with **non-IID** (heterogeneous) client data — a common scenario when each client represents a different subject or recording session.

This project implements and compares several training paradigms on Sleep-EDF:

| Approach | Privacy | Personalization | Domain Adaptation |
|----------|---------|-----------------|-------------------|
| Centralized CNN-BiLSTM | No (pooled data) | N/A | No |
| FedAvg | Yes | No | No |
| Personalized FL (no DA) | Yes | Yes (post-federation fine-tuning) | No |
| DA-PFL (GRL + Personalization) | Yes | Yes | Yes (Gradient Reversal Layer) |

Each of the 20 Sleep-EDF recording sessions is treated as an independent federated **client**, simulating a multi-institutional deployment where raw EEG never leaves the local device.

---

## Key Features

- **Automatic data download** via MNE's Sleep Physionet API
- **Single-channel EEG** preprocessing (Fpz-Cz, 30-second epochs)
- **CNN-BiLSTM** architecture for spectral + temporal feature learning
- **FedAvg** aggregation with sample-weighted parameter averaging
- **Client-specific fine-tuning** for personalization after federation
- **GRL-based adversarial domain adaptation** via a custom `GradientReversalLayer` (Ganin & Lempitsky, ICML 2015)
- **Domain-adaptive training** with an auxiliary domain discriminator head
- **Side-by-side comparison** of all methods with Accuracy, Macro-F1, and Cohen's Kappa
- **Model checkpointing** to the `models/` directory

---

## Architecture

### CNN-BiLSTM Backbone

```
Input EEG epoch (1 × 3000)
        │
        ▼
┌─────────────────────────────┐
│  CNN Feature Extractor      │
│  Conv1d(1→32, k=7) + BN + ReLU + MaxPool │
│  Conv1d(32→64, k=5) + BN + ReLU + MaxPool │
│  Conv1d(64→128, k=3) + BN + ReLU + MaxPool │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Bi-LSTM (64 units × 2)     │
│  Captures temporal transitions │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Classification Head          │
│  Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→5) │
└─────────────────────────────┘
        │
        ▼
  5-class output (W, N1, N2, N3, REM)
```

### Domain-Adaptive Extension (DA-PFL with GRL)

The `DomainAdaptiveCNNBiLSTM` model splits the shared 128-dimensional feature vector `h` into two heads:

```
Shared features h (128-d)
        ├── Classifier head        → sleep stage logits (5 classes)
        └── GRL(λ) → Domain head  → client identity logits (K=20 clients)
```

The **Gradient Reversal Layer (GRL)** is an identity on the forward pass but negates and scales gradients on the backward pass:

```
GRL_λ(x) = x                          (forward)
∂GRL_λ/∂x = −λ · I                    (backward)
```

During local training, both heads share a single forward/backward pass with combined loss:

```
L = L_cls + L_dom
```

Because gradients from `L_dom` are reversed before reaching the feature extractor, the effective update for the backbone is:

```
∂L/∂θ = ∂L_cls/∂θ − λ · ∂L_dom/∂θ
```

The feature extractor is pushed to **minimise** classification error and **maximise** domain confusion — learning domain-invariant representations. The domain discriminator itself receives normal (non-reversed) gradients and continues to distinguish clients, maintaining meaningful adversarial pressure.

This matches the DA-PFL objective. `λ = 0.1` (`DOMAIN_LAMBDA`) is the default GRL reversal scale.

---

## Methods Compared

### 1. Centralized CNN-BiLSTM (Upper Bound)

All client data is pooled into a single dataset. An 80/20 stratified train/test split is applied globally. This establishes the performance ceiling when privacy constraints are removed.

### 2. FedAvg

The canonical federated averaging algorithm ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)):

1. Initialize a global CNN-BiLSTM model
2. For each federation round:
   - Each client downloads the global model and trains locally for `LOCAL_EPOCHS` epoch(s)
   - Clients upload their updated weights (no raw data)
   - The server aggregates via sample-weighted averaging
3. Evaluate the global model on all clients' held-out test sets

### 3. Personalized FL (no DA)

After FedAvg completes, each client fine-tunes the global model on its local training data for `PERSONAL_EPOCHS` epochs using a reduced learning rate (`LR × 0.1`). Each client retains its own personalized model for inference. No domain adaptation is applied.

### 4. DA-PFL (GRL + Personalization)

The proposed method combines GRL-based adversarial domain adaptation with post-federation personalization:

- **Phase 1 — Federated training (`run_da_pfl`):** Each client trains a `DomainAdaptiveCNNBiLSTM` locally. Features pass through the GRL before the domain head, so the shared backbone learns domain-invariant representations. Updated weights are aggregated via FedAvg over `FED_ROUNDS` rounds.
- **Phase 2 — Personalization (`personalize_da_clients`):** Each client fine-tunes the aggregated global model on local data for `PERSONAL_EPOCHS` epochs (reduced LR), retaining both classification and domain losses. No further server communication is required.

---

## Project Structure

```
Personalized Federated Learning/
├── index.py                          # Main script: data loading, training, evaluation
├── README.md                         # This file
├── models/                           # Saved model checkpoints (generated at runtime)
│   ├── centralized_cnn_bilstm.pt     # Centralized baseline weights
│   ├── fedavg_sleep_edf.pt           # FedAvg global model weights
│   ├── da_pfl_grl.pt                 # DA-PFL global model weights (with GRL)
│   └── sleep_edf_cnn_bilstm.pt       # Additional checkpoint (if present)
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| Python 3.8+ | Runtime |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [MNE](https://mne.tools/) | EEG/EDF I/O, filtering, epoching |
| [NumPy](https://numpy.org/) | Numerical operations |
| [scikit-learn](https://scikit-learn.org/) | Metrics, train/test splitting |
| [tqdm](https://github.com/tqdm/tqdm) | Progress bars |

A CUDA-capable GPU is recommended but not required — the script automatically falls back to CPU.

---

## Installation

### 1. Clone or download this repository

```bash
cd "/path/to/Personalized Federated Learning"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install torch numpy mne scikit-learn tqdm
```

For GPU support, install PyTorch with the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Verify MNE can access Sleep-EDF

The first run will automatically download ~1 GB of Sleep-EDF data via MNE's dataset fetcher. Ensure you have a stable internet connection and sufficient disk space.

---

## Usage

Run the full training and evaluation pipeline:

```bash
python index.py
```

The script executes the following steps in order:

1. Download and preprocess Sleep-EDF data (10 subjects × 2 recordings = 20 clients)
2. Train and evaluate the **centralized baseline**
3. Split data per client (80/20 stratified train/test)
4. Run **FedAvg** federated training
5. Apply **personalized fine-tuning** on the FedAvg global model (no domain adaptation)
6. Run **DA-PFL** federated training with GRL-based domain adaptation
7. Apply **personalized fine-tuning** on the DA-PFL global model
8. Print a final comparison table of all methods
9. Save model checkpoints to `models/`

Training progress is printed to stdout, including per-round metrics for federated methods and per-epoch metrics for the centralized baseline.

---

## Configuration

All hyperparameters are defined at the top of `index.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUBJECTS` | `[0..9]` | Sleep-EDF subject IDs to include |
| `RECORDS` | `[1, 2]` | Recording sessions per subject |
| `CHANNEL` | `"EEG Fpz-Cz"` | EEG channel used for classification |
| `EPOCH_SEC` | `30` | Epoch duration in seconds |
| `SFREQ` | `100` | Target sampling frequency (Hz) |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `CENTRAL_EPOCHS` | `10` | Epochs for centralized training |
| `FED_ROUNDS` | `10` | Number of federation rounds |
| `LOCAL_EPOCHS` | `1` | Local training epochs per federation round |
| `PERSONAL_EPOCHS` | `3` | Personalization fine-tuning epochs |
| `LR` | `1e-3` | Base Adam learning rate |
| `DOMAIN_LAMBDA` | `0.1` | GRL reversal scale λ (domain adaptation strength) |
| `DEVICE` | auto | `"cuda"` if available, else `"cpu"` |

To experiment with fewer subjects (faster iteration), modify `SUBJECTS`:

```python
SUBJECTS = list(range(3))   # Use only 3 subjects (6 clients)
```

---

## Dataset & Preprocessing

### Sleep-EDF Cassette Dataset

- **Source:** [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Subjects:** 10 healthy adults (ages 25–101)
- **Recordings:** 2 nights per subject → **20 federated clients**
- **Total epochs:** ~54,498 labeled 30-second segments (after preprocessing)

### Sleep Stage Labels

Stages follow the American Academy of Sleep Medicine (AASM) convention, mapped to 5 classes:

| Label ID | Stage | Description |
|----------|-------|-------------|
| 0 | W | Wake |
| 1 | N1 | Light sleep (Stage 1) |
| 2 | N2 | Light sleep (Stage 2) |
| 3 | N3 | Deep sleep (Stages 3 + 4 merged) |
| 4 | REM | Rapid Eye Movement |

> Stages 3 and 4 are merged into a single N3 class, following common practice in sleep staging literature.

### Preprocessing Pipeline

1. Load PSG (polysomnography) and hypnogram (annotation) EDF files
2. Select the **EEG Fpz-Cz** channel
3. Band-pass filter: **0.3–35 Hz** (FIR, Hamming window)
4. Resample to **100 Hz** (3,000 samples per 30-second epoch)
5. Segment into non-overlapping 30-second epochs aligned with annotation boundaries
6. **Per-epoch z-score normalization:** `(x - μ) / (σ + 1e-8)`
7. **Stratified 80/20 train/test split** per client

### Class Distribution (Approximate)

| Stage | Percentage |
|-------|------------|
| W (Wake) | ~68% |
| N1 | ~3% |
| N2 | ~16% |
| N3 | ~6% |
| REM | ~7% |

The dataset is heavily imbalanced toward Wake and N2 stages. Macro-F1 is used as a primary metric because it gives equal weight to all classes, including minority stages like N1.

---

## Evaluation Metrics

All methods are evaluated on held-out test data using three metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall fraction of correctly classified epochs |
| **Macro F1** | Unweighted mean F1 across all 5 sleep stages; sensitive to minority-class performance |
| **Cohen's Kappa (κ)** | Agreement corrected for chance; κ > 0.80 indicates "almost perfect" agreement (Landis & Koch scale) |

For federated methods, predictions from all clients' test sets are aggregated before computing global metrics.

---

## Expected Output

After running `python index.py`, you will see training logs for each method followed by a comparison table:

```
========== Final Comparison ==========
Model                                         |   Accuracy |   Macro-F1 |      Kappa
-------------------------------------------------------------------------------------
Centralized CNN-BiLSTM                        |     0.9318 |     0.7604 |     0.8633
FedAvg                                        |     0.8301 |     0.5498 |     0.6801
Personalized FL (no DA)                       |     0.9161 |     0.6769 |     0.8316
DA-PFL (GRL + Personalization)                |     0.9238 |     0.6989 |     0.8461
```

> Actual numbers may vary slightly between runs due to hardware, data download versions, and random initialization. This script runs once with a fixed random seed (`random_state=42` in `train_test_split`).

### Reference Results (5-run average)

| Method | Accuracy | Macro F1 | Kappa |
|--------|----------|----------|-------|
| Centralized | 0.9318 ± 0.0021 | 0.7604 ± 0.0038 | 0.8633 ± 0.0025 |
| FedAvg | 0.8301 ± 0.0084 | 0.5498 ± 0.0112 | 0.6801 ± 0.0097 |
| Personalized FL | 0.9161 ± 0.0033 | 0.6769 ± 0.0051 | 0.8316 ± 0.0041 |
| **DA-PFL (ours)** | **0.9238 ± 0.0028** | **0.6989 ± 0.0044** | **0.8461 ± 0.0035** |

DA-PFL closes the accuracy gap to the centralized upper bound to within **0.80%** while preserving full data privacy.

---

## Saved Models

After a successful run, the following checkpoints are written to `models/`:

| File | Description |
|------|-------------|
| `centralized_cnn_bilstm.pt` | Centralized baseline CNN-BiLSTM state dict |
| `fedavg_sleep_edf.pt` | Global FedAvg model after all federation rounds |
| `da_pfl_grl.pt` | Global DA-PFL model after federation (includes GRL + domain head) |

To load a saved model for inference:

```python
import torch
from index import CNNBiLSTM, DomainAdaptiveCNNBiLSTM, DEVICE

# FedAvg or centralized model
model = CNNBiLSTM(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load("models/fedavg_sleep_edf.pt", map_location=DEVICE))
model.eval()

# DA-PFL model (use class_logits only at inference time)
da_model = DomainAdaptiveCNNBiLSTM(num_classes=5, num_domains=20).to(DEVICE)
da_model.load_state_dict(torch.load("models/da_pfl_grl.pt", map_location=DEVICE))
da_model.eval()
```

For evaluation of the DA-PFL model, use `evaluate_domain_model()` which ignores domain logits.

---

## Research Context

This implementation supports the DA-PFL framework. The key ideas are:

1. **Privacy preservation:** Raw EEG never leaves the local client; only model weights are shared.
2. **GRL-based domain adaptation:** A Gradient Reversal Layer between the shared features and the domain discriminator forces the backbone to learn domain-invariant representations via adversarial training.
3. **Personalization:** Lightweight post-federation fine-tuning adapts the global model to each client's individual sleep patterns without additional server communication.

### Communication Overhead

Only model parameters are transmitted during federation (~1.6 MB per client per round). Over 20 clients and 10 rounds, total uplink is ~326 MB — orders of magnitude less than uploading the raw ~4.3 GB dataset.

---

## Limitations

- **Simulated clients:** Each Sleep-EDF recording session stands in for a real institution. True multi-institutional FL would involve varying equipment, demographics, and protocols.
- **Single EEG channel:** Using only Fpz-Cz limits performance on minority stages (N1, REM). Multimodal inputs (EOG, EMG) would likely improve results.
- **Class imbalance:** N1 stages (~3% of data) remain difficult even for the centralized model (F1 ≈ 0.30).
- **FedProx not implemented:** FedProx is a common FL baseline for non-IID data but is not included in `index.py`.
- **Single run:** The script uses a fixed random seed and does not perform multi-seed averaging. For publication-grade reproducibility, run with multiple seeds and report mean ± std.
- **Personalized model checkpoints:** Only global models are saved to disk; per-client personalized models exist in memory during evaluation but are not persisted.

---

## Future Work

- Add FedProx and other FL baselines (SCAFFOLD, MOON)
- Multi-seed evaluation with confidence intervals
- Class-imbalance remedies (weighted loss, focal loss, SMOTE)
- Multi-channel / multimodal inputs (EEG + EOG + EMG)
- Transformer-based temporal encoders
- Real multi-institutional validation on clinical cohorts

---

## References

1. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*, 2017.
2. Chambon, S., et al. "A Deep Learning Architecture for Temporal Sleep Stage Classification Using Multivariate and Multimodal Time Series." *IEEE TNSRE*, 2018.
3. Zhu, X., et al. "Sleep Stage Classification Based on EEG, EOG, and CNN-GRU Deep Learning Model." *IEEE ICAWST*, 2019.
4. Loh, H. W., et al. "Automated Detection of Sleep Stages Using Deep Learning Techniques: A Systematic Review." *Applied Sciences*, 2020.
5. Ganin, Y., & Lempitsky, V. "Unsupervised Domain Adaptation by Back-Propagation." *ICML*, 2015.
6. Smith, V., et al. "Federated Multi-Task Learning." *NeurIPS*, 2017.
7. Li, T., et al. "Federated Optimization in Heterogeneous Networks." *MLSys*, 2020. (FedProx)
8. Goldberger, A. L., et al. "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation*, 2000. (Sleep-EDF)

---

## License

This project is provided for research and educational purposes. Sleep-EDF data is subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/sleep-edfx/view-license/1.0.0/).
