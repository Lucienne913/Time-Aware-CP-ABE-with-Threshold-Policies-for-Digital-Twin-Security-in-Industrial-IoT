# T-CP-ABE: Time-Aware CP-ABE for Digital Twin Network Security

This repository contains the academic implementation of **T-CP-ABE**, a Time-aware Ciphertext-Policy Attribute-Based Encryption framework designed for Industrial IoT Digital Twin security.

## Publication

This work is submitted to **IEEE Internet of Things Journal** (SCI Q1, Impact Factor: 8.2).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Digital Twin Security Framework                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  IoT Devices │───▶│  Digital     │───▶│  Cloud Services      │  │
│  │  (Sensors,   │    │  Twin        │    │  (Eclipse Ditto)     │  │
│  │  Actuators)  │    │  Manager     │    │                      │  │
│  └──────┬───────┘    └──────┬───────┘    └─────────┬────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  BLS         │    │  T-CP-ABE    │    │  Threat Diffusion   │  │
│  │  Signature   │    │  Engine      │    │  Model (DDPM)       │  │
│  │  Authentication│  │  (Time +     │    │  (Anomaly Detection)│  │
│  │              │    │  Threshold)  │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Time-Aware Access Control**: Hash-chain based time predicate with constant-time verification
- **k-n Threshold Policy**: Flexible access policies supporting THRESHOLD gates
- **Batch Revocation**: Efficient attribute revocation with sub-0.1ms latency
- **Dual Access Mode**: Online (real-time) and offline (cached token) access
- **Threat Awareness**: Diffusion-based anomaly detection for closed-loop security
- **Digital Twin Integration**: Eclipse Ditto compatible device shadow management

## Directory Structure

```
academic_implementation/
├── src/                      # Core source code
│   ├── t_cp_abe.py           # T-CP-ABE main implementation
│   ├── setup.py              # System initialization (bilinear pairing)
│   ├── auth.py               # Bidirectional authentication
│   ├── signatures.py         # BLS signatures and device certificates
│   ├── digital_twin.py       # Digital Twin Manager
│   ├── diffusion.py          # Threat diffusion model (DDPM)
│   ├── subprocess_worker.py  # OOM protection worker
│   ├── train_diffusion.py    # Model training script
│   ├── baselines/            # Baseline implementations
│   │   ├── bsw07_abe.py      # BSW07 CP-ABE baseline
│   │   ├── lightweight_abe.py# BSW07 AND-gate only
│   │   ├── guo2024_baseline.py# Guo2024 (simulated)
│   │   └── zhang2024_baseline.py# Zhang2024 (simulated)
│   └── weights/              # Model weights (excluded from Git)
├── tests/                    # Test suite (20+ test files)
│   ├── conftest.py           # pytest fixtures
│   ├── test_basic.py         # Basic functionality tests
│   ├── test_security.py      # Security tests (IND-CPA, EUF-CMA)
│   ├── test_performance.py   # Performance tests
│   ├── test_threshold.py     # Threshold policy tests
│   ├── test_sota_comparison.py# SOTA comparison tests
│   └── ...                   # Additional test files
├── experiments/              # Experiment scripts
│   ├── generate_paper_figures.py # Paper figure generation
│   ├── preprocess_unsw_nb15.py   # UNSW-NB15 dataset preprocessing
│   └── results/              # Experiment results (JSON)
├── paper/                    # Paper artifacts
│   ├── paper.tex             # LaTeX source
│   ├── paper.pdf             # Compiled PDF
│   ├── literature.bib        # Bibliography
│   └── figures/              # Paper figures (PDF/PNG)
├── Dockerfile                # Docker build instructions
├── pytest.ini                # pytest configuration
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.9 (Charm-Crypto compatibility requirement)
- Charm-Crypto (requires PBC library)

### Docker Installation (Recommended)

```bash
docker build -t tcabe-academic .
docker run -it --rm tcabe-academic
```

### Local Installation

```bash
pip install -r requirements.txt
```

**Note**: Charm-Crypto requires PBC library installation. See Dockerfile for detailed instructions.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -v -m basic
python -m pytest tests/ -v -m security
python -m pytest tests/ -v -m performance

# Run slow tests (excluded by default)
python -m pytest tests/ -v -m slow
```

## Test Categories

| Marker | Description |
|--------|-------------|
| `basic` | Basic functionality tests |
| `security` | Security tests (IND-CPA, EUF-CMA) |
| `performance` | Performance tests |
| `edge_case` | Edge case tests |
| `memory` | Memory leak tests |
| `comparison` | Comparison tests |
| `slow` | Slow tests (>30 seconds) |
| `embedded` | Embedded device tests |
| `integration` | Integration tests |

## Generating Paper Figures

```bash
python experiments/generate_paper_figures.py
```

This generates all figures for the IEEE IoT Journal submission.

## Key Components

### T-CP-ABE Core

- **Setup**: Bilinear pairing group initialization (SS1024 curve, 128-bit security)
- **KeyGen**: User secret key generation with time predicate support
- **Encrypt**: Ciphertext generation with policy embedding
- **Decrypt**: Attribute-based decryption with time token verification

### Hash Chain Time Predicate

- Merkle hash chain for time token generation
- Constant-time verification using checkpoint optimization
- Resistance to time token forgery attacks

### Threat Diffusion Model

- Denoising Diffusion Probabilistic Model (DDPM)
- Attribute-level anomaly detection
- Closed-loop security integration

## Security Model

- IND-sCPA-T: Indistinguishability under Selective CPA with Temporal constraints
- Reduces to DBDH assumption (Decisional Bilinear Diffie-Hellman)
- Collusion resistance with threshold-revocation mechanism

## License

This project is for academic research purposes only.

## Contact

For questions or issues, please contact the corresponding author.
