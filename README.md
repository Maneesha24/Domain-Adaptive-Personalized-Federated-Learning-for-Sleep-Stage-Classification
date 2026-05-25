# Personalized Federated Learning for Sleep Stage Classification

**Isha Venigalla** — Software Development Engineer II, Amazon · New York City, USA

PyTorch implementation of **Domain-Adaptive Personalized Federated Learning (DA-PFL)** for sleep stage classification on [Sleep-EDF Cassette](https://physionet.org/content/sleep-edfx/1.0.0/). Compares centralized training, FedAvg, personalized FL, and GRL-based domain-adaptive FL with a shared **CNN-BiLSTM** backbone.

**Conference Paper:** [`Personalized_Federated_Learning_for_Sleep_Stage_Classification.pdf`](Personalized_Federated_Learning_for_Sleep_Stage_Classification.pdf)

---

## Quick Start

```bash
cd "/path/to/Personalized Federated Learning"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python index.py
```

Default run uses seeds `42`, `52`, `62` and writes artifacts to `paper_results/`. First launch downloads ~1 GB of Sleep-EDF data via MNE.

```bash
# Faster iteration: skip centralized baseline, verbose logging
python index.py --skip-centralized --verbose

# Custom seeds / output directory
python index.py --seeds 42 52 62 72 82 --out-dir my_run
```

---

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Saved Models & Inference](#saved-models--inference)
- [Code Documentation](#code-documentation)
- [Research Context](#research-context)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [License](#license)

---

## Overview

Sleep stage classification from EEG supports diagnosis of sleep disorders, but centralized deep learning requires pooling sensitive recordings. **Federated learning (FL)** trains collaboratively without sharing raw data, yet **FedAvg** often underperforms when clients are **non-IID** — e.g., each client is a different subject or night.

| Approach | Privacy | Personalization | Domain adaptation |
|----------|---------|-----------------|-------------------|
| Centralized CNN-BiLSTM | No (pooled data) | — | No |
| FedAvg | Yes | No | No |
| Personalized FL (no DA) | Yes | Post-FL fine-tuning | No |
| **DA-PFL (ours)** | Yes | Post-FL fine-tuning | GRL + domain head |

**20 federated clients** — 10 Sleep-EDF subjects × 2 nights per subject. Each recording is one client; raw EEG never leaves the client during training.

All experiments run through **`index.py`**: multi-seed evaluation, checkpoints, CSV/LaTeX export, convergence curves, and per-class reports. The codebase uses detailed NumPy-style docstrings throughout `index.py`.

---

## Methods

### 1. Centralized CNN-BiLSTM (upper bound)

All data pooled; global 80/20 stratified train/test split. Performance ceiling without privacy constraints.

### 2. FedAvg

Standard federated averaging ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)): each round, clients train locally for `LOCAL_EPOCHS`, upload weights, server applies sample-weighted averaging. One global model evaluated on all clients' test sets.

### 3. Personalized FL (no DA)

After FedAvg, each client fine-tunes the global model for `PERSONAL_EPOCHS` at `LR × PERSONAL_LR_FACTOR`. Classification loss only; no domain head.

### 4. DA-PFL (GRL + personalization)

- **Phase 1 (`run_da_pfl_phase1`):** `DomainAdaptiveCNNBiLSTM` with loss `L_cls + L_dom`. GRL reverses domain gradients into the backbone so features become domain-invariant. FedAvg over `FED_ROUNDS`.
- **Phase 2 (`run_da_pfl`):** Each client fine-tunes with **classification loss only** (domain logits ignored). Matches the paper protocol: personalization without continued adversarial domain training.

---

## Architecture

### CNN-BiLSTM backbone

```
Input (1 × 3000) → 3× Conv1d blocks (BN, ReLU, MaxPool)
                 → BiLSTM (64×2, last hidden state 128-d)
                 → MLP → 5 classes (W, N1, N2, N3, REM)
```

### DA-PFL extension

```
Backbone features h (128-d)
  ├─ Classifier → sleep stage (5 classes)
  └─ GRL(λ) → Domain head → client ID (20 domains)
```

GRL forward is identity; backward scales gradients by `−λ`:

```
L = L_cls + L_dom   →   ∂L/∂θ_backbone = ∂L_cls/∂θ − λ · ∂L_dom/∂θ
```

Default `λ = DOMAIN_LAMBDA = 0.1`.

---

## Project Structure

```
Personalized Federated Learning/
├── index.py                              # Main experiment script
├── README.md
├── requirements.txt
├── Personalized_Federated_Learning_for_Sleep_Stage_Classification.pdf
└── paper_results/                        # Created by index.py (default --out-dir)
    ├── experiment_metadata.json
    ├── per_seed_results.csv              # After full multi-seed run
    ├── summary_mean_std.csv
    ├── convergence_curves.csv
    ├── delta_rows.csv
    ├── latex_table_rows.txt
    └── seed_<N>/
        ├── centralized.pt, fedavg.pt, da_pfl_phase1_grl.pt
        ├── pers_fl_client_<id>.pt, da_pfl_client_<id>.pt
        └── *_report.txt, *_report.csv
```

> **Note:** Checkpoint directories under `paper_results/seed_*` are large (~hundreds of MB per seed). Consider adding `paper_results/` or `*.pt` to `.gitignore` if you do not version trained weights.

---

## Requirements & Installation

| Package | Purpose |
|---------|---------|
| Python 3.8+ | Runtime |
| PyTorch | Models and training |
| MNE | Sleep-EDF download, filtering, epoching |
| NumPy, pandas | Arrays and result tables |
| scikit-learn | Metrics and stratified splits |
| tqdm | Progress bars |

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install a [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) build for GPU training. The script uses `"cuda"` when available, otherwise `"cpu"`.

---

## Usage

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | `42 52 62` | Random seeds (space-separated) |
| `--out-dir` | `paper_results` | Output root for CSVs and checkpoints |
| `--skip-centralized` | off | Skip centralized upper bound |
| `--verbose` | off | Per-epoch / per-round logging |

**Per seed**, `run_all_experiments` in `index.py`:

1. Loads Sleep-EDF (20 clients)
2. Stratified 80/20 split per client
3. Centralized baseline (optional)
4. FedAvg → Personalized FL
5. DA-PFL Phase 1 → Phase 2
6. Saves checkpoints and classification reports under `seed_<N>/`

**After all seeds:** writes aggregate CSVs, LaTeX rows, delta table, and prints the multi-seed summary.

---

## Configuration

Edit constants at the top of `index.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUBJECTS` | `0..9` | Sleep-EDF subjects |
| `RECORDS` | `[1, 2]` | Nights per subject |
| `CHANNEL` | `EEG Fpz-Cz` | Input channel |
| `EPOCH_SEC` / `SFREQ` | `30` / `100` | Epoch length; 3000 samples/epoch |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `CENTRAL_EPOCHS` | `10` | Centralized training epochs |
| `FED_ROUNDS` | `10` | Federation rounds |
| `LOCAL_EPOCHS` | `1` | Local epochs per FedAvg round |
| `PERSONAL_EPOCHS` | `3` | Personalization epochs |
| `LR` | `1e-3` | Adam learning rate |
| `PERSONAL_LR_FACTOR` | `0.1` | Personalization LR scale |
| `DOMAIN_LAMBDA` | `0.1` | GRL strength λ |
| `DEFAULT_SEEDS` | `[42, 52, 62]` | Overridden by `--seeds` |

Smaller ablation: `SUBJECTS = list(range(3))` → 6 clients, faster runs.

---

## Dataset & Preprocessing

- **54,498** epochs after preprocessing (10 subjects × 2 nights × ~2.7k epochs/night)
- **Labels:** W, N1, N2, N3 (S3+S4 merged), REM — see `ANNOTATION_DESC_2_ID` in `index.py`
- **Pipeline:** load EDF → Fpz-Cz → 0.3–35 Hz band-pass → resample 100 Hz → 30 s epochs → per-epoch z-score → stratified client split

| Stage | Count | Share |
|-------|-------|-------|
| W | 37,115 | 68% |
| N1 | 1,689 | 3% |
| N2 | 8,927 | 16% |
| N3 | 3,074 | 6% |
| REM | 3,693 | 7% |

Macro-F1 is emphasized because accuracy is dominated by Wake/N2.

---

## Evaluation Metrics

| Metric | Role |
|--------|------|
| **Accuracy** | Overall correct epochs |
| **Macro F1** | Equal weight per stage; highlights N1/REM |
| **Cohen's κ** | Chance-corrected agreement |

Federated metrics pool predictions across all clients' **test** sets before scoring.

---

## Results

### Multi-seed summary (seeds 42, 52, 62)

Derived from pooled test-set classification reports in `paper_results/seed_*/`. Re-run `python index.py` to refresh and populate `summary_mean_std.csv` with κ and exact aggregates.

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| Centralized CNN-BiLSTM | 0.9300 ± 0.0000 | 0.7333 ± 0.0208 |
| FedAvg | 0.8267 ± 0.0289 | 0.5400 ± 0.0300 |
| Pers. FL (no DA) | 0.9167 ± 0.0058 | 0.6800 ± 0.0000 |
| **DA-PFL (ours)** | **0.9167 ± 0.0058** | **0.6800 ± 0.0100** |

DA-PFL reaches the same pooled accuracy as Personalized FL while Phase 1 learns domain-invariant representations via GRL. FedAvg lags on macro-F1 due to non-IID client drift. Centralized training remains the upper bound (~93% accuracy).

### Example console output

```
================================================================================
FINAL MULTI-SEED SUMMARY
================================================================================
                  method  accuracy_mean  accuracy_std  macro_f1_mean  macro_f1_std  kappa_mean  kappa_std  n_seeds
...
```

LaTeX table rows: `paper_results/latex_table_rows.txt` · DA-PFL deltas: `paper_results/delta_rows.csv`

---

## Saved Models & Inference

| Checkpoint | Description |
|------------|-------------|
| `centralized.pt` | Pooled-data upper bound |
| `fedavg.pt` | Global FedAvg weights |
| `da_pfl_phase1_grl.pt` | Global model after Phase 1 |
| `pers_fl_client_<id>.pt` | Personalized FL, client `id` |
| `da_pfl_client_<id>.pt` | DA-PFL after Phase 2 |

```python
import torch
from index import CNNBiLSTM, DomainAdaptiveCNNBiLSTM, DEVICE, evaluate_domain_model

SEED = 42
path = f"paper_results/seed_{SEED}"

# FedAvg global model
model = CNNBiLSTM(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load(f"{path}/fedavg.pt", map_location=DEVICE, weights_only=True))
model.eval()

# DA-PFL personalized client 0
client = DomainAdaptiveCNNBiLSTM(num_classes=5, num_domains=20, lam=0.1).to(DEVICE)
client.load_state_dict(torch.load(f"{path}/da_pfl_client_0.pt", map_location=DEVICE, weights_only=True))
client.eval()
```

For DA-PFL evaluation, use `evaluate_domain_model()` so only sleep-stage logits are used.

---

## Code Documentation

`index.py` documents the full pipeline with NumPy-style docstrings (parameters, returns, notes, cross-references).

| Symbol | Role |
|--------|------|
| `load_sleep_edf`, `split_clients` | Data loading and partitioning |
| `CNNBiLSTM`, `DomainAdaptiveCNNBiLSTM`, `GradientReversalLayer` | Models |
| `run_centralized`, `run_fedavg`, `run_personalized_fl` | Baselines |
| `run_da_pfl_phase1`, `run_da_pfl` | DA-PFL phases |
| `run_all_experiments`, `parse_args` | CLI entry |

```bash
python -c "import index; help(index.run_da_pfl_phase1)"
```

---

## Research Context

1. **Privacy:** Only model weights cross the network during federation.
2. **Domain invariance:** GRL encourages features that confuse the domain (client) classifier while preserving sleep-stage signal.
3. **Personalization:** Short local fine-tuning adapts to subject/night-specific patterns without extra server rounds.

**Communication cost:** ~1.6 MB per client per round × 20 clients × 10 rounds ≈ 326 MB of weight updates vs. ~4.3 GB of raw EEG.

---

## Limitations & Future Work

**Limitations:** Simulated clients (not true multi-site cohorts); single EEG channel; severe N1 imbalance; FedProx and other FL baselines not implemented; multi-seed runs are CPU/GPU intensive.

**Future work:** FedProx / SCAFFOLD / MOON baselines; class-weighted or focal loss; multimodal inputs; transformers; clinical multi-site validation; CLI flags for `FED_ROUNDS` and hyperparameter sweeps.

---

## References

1. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
2. Ganin & Lempitsky, *Unsupervised Domain Adaptation by Back-Propagation*, ICML 2015.
3. Chambon et al., *IEEE TNSRE*, 2018 — deep learning sleep staging.
4. Goldberger et al., *Circulation*, 2000 — PhysioNet / Sleep-EDF.
5. Li et al., *FedProx*, MLSys 2020.

---

## License

Research and educational use. Sleep-EDF is under the [PhysioNet license](https://physionet.org/content/sleep-edfx/view-license/1.0.0/).
