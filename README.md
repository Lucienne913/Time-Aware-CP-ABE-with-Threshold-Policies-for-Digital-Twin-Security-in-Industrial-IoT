# T-CP-ABE: Time-Aware CP-ABE for Digital Twin Network Security

Academic implementation of **T-CP-ABE**, a Time-aware Ciphertext-Policy Attribute-Based Encryption framework for Industrial IoT Digital Twin security.

## Publication

Submitted to **IEEE Internet of Things Journal** (SCI Q1, Impact Factor: 8.2).

## System Architecture

See [`paper/figures/fig01_system_architecture.pdf`](paper/figures/fig01_system_architecture.pdf) for the complete system diagram.

**Core Components:**

| Layer | Components |
|-------|------------|
| Device Layer | IoT sensors/actuators, BLS signature authentication |
| Digital Twin | Eclipse Ditto-compatible device shadow management |
| Access Control | T-CP-ABE engine (time predicates + threshold policies) |
| Threat Detection | DDPM-based anomaly detection (closed-loop) |

## Key Features

- **Time-Aware Access Control**: Hash-chain based time predicate with constant-time verification
- **k-n Threshold Policy**: Native THRESHOLD gate support
- **Batch Revocation**: Sub-0.1ms latency per attribute
- **Dual Access Mode**: Online (real-time) and offline (cached token) access
- **Threat Awareness**: Diffusion-based anomaly detection
- **Digital Twin Integration**: Eclipse Ditto compatible

## Directory Structure

```
academic_implementation/
├── src/                      # Core source code
│   ├── t_cp_abe.py           # T-CP-ABE implementation
│   ├── setup.py              # Bilinear pairing setup
│   ├── auth.py               # Bidirectional authentication
│   ├── signatures.py         # BLS signatures
│   ├── digital_twin.py       # Digital Twin Manager
│   ├── diffusion.py          # Threat detection model
│   ├── baselines/            # BSW07, Lightweight ABE, etc.
│   └── weights/              # Pre-trained model weights
├── tests/                    # Test suite (20+ files)
├── experiments/              # Scripts and results
│   ├── generate_paper_figures.py
│   ├── preprocess_unsw_nb15.py
│   └── results/              # Experiment data (JSON)
├── paper/                    # Paper materials
│   ├── paper.tex             # LaTeX source
│   ├── paper.pdf             # Compiled paper
│   ├── literature.bib        # Bibliography
│   └── figures/              # System diagrams
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.9
- Charm-Crypto (requires PBC library)

### Docker (Recommended)

```bash
docker build -t tcabe-academic .
docker run -it --rm tcabe-academic
```

### Local Installation

```bash
pip install -r requirements.txt
```

See [Dockerfile](Dockerfile) for PBC library installation details.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run by category
python -m pytest tests/ -v -m security
python -m pytest tests/ -v -m performance
```

## Test Categories

| Marker | Description |
|--------|-------------|
| `basic` | Basic functionality |
| `security` | IND-CPA, EUF-CMA proofs |
| `performance` | Timing benchmarks |
| `scalability` | Large-scale tests |
| `embedded` | Embedded device simulation |

## Generate Paper Figures

```bash
python experiments/generate_paper_figures.py
```

Outputs to `paper/figures/`.

## Security Model

- **IND-sCPA-T**: Indistinguishability under Selective CPA with Temporal queries
- **DBDH Reduction**: Security reduces to Decisional Bilinear Diffie-Hellman
- **Collusion Resistance**: Threshold-revocation mechanism

## Performance Highlights

| Operation | Time (1000 attrs) |
|-----------|-------------------|
| KeyGen | 15.2 ms |
| Encrypt | 8.7 ms |
| Decrypt | 12.3 ms |
| Batch Revocation | 0.08 ms/attr |

## License

Academic research use only.

## Contact

For questions, contact the corresponding author.
