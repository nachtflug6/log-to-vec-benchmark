# Log-to-Vec: Benchmarking AI Models for Log Embeddings

A comprehensive benchmarking framework for investigating and evaluating AI models that transform time-stamped log files into vector representations (embeddings). This project focuses on unsupervised and self-supervised learning approaches for capturing behavioral patterns in log sequences.

Start here: [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) explains why log embeddings matter, what constraints we care about, and why we compare models under different deployment limits (e.g., edge settings).

## Project Overview

Complex software-controlled systems (e.g., CPS and production systems) continuously generate large volumes of log data. This project provides tools and benchmarks to:

- Transform log sequences into meaningful embeddings
- Compare different embedding approaches (autoencoders, Transformers, contrastive learning)
- Evaluate embeddings on downstream tasks (similarity search, clustering, anomaly detection)
- Study unsupervised training strategies for log data

## Student Quick Start (Where to Add What)

If you're adding new methods or experiments, these are the main places to work:

- Models: add your model class in [src/log_to_vec/models](src/log_to_vec/models)
- Training logic: extend/adjust trainers in [src/log_to_vec/training](src/log_to_vec/training)
- Data handling: parsing/datasets in [src/log_to_vec/data](src/log_to_vec/data)
- Evaluation metrics: add new metrics in [src/log_to_vec/evaluation/metrics.py](src/log_to_vec/evaluation/metrics.py)
- Example scripts: add runnable demos in [examples](examples)
- Experiment automation: place hyperparameter sweeps in [experiments](experiments)

Detailed guidance for students is in [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

## Features

- **Data Generation**: Realistic PLC-like log generator for testing
- **Multiple Model Architectures**: LSTM autoencoders, Transformer autoencoders, contrastive models
- **Comprehensive Evaluation**: Metrics for reconstruction, similarity, clustering, and retrieval
- **Mode-Change Baseline**: Optional unsupervised change-point and segment clustering scaffold for numerical time-series logs
- **Structured Experiment Registry**: Schema and templates for reproducible run metadata
- **Flexible Configuration**: YAML-based configuration system
- **Containerization**: Docker and Apptainer support for reproducibility and HPC deployment

## Quick Start

### Installation

#### Option 1: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd log-to-vec

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Option 2: Docker

```bash
# Build the Docker image
docker build -t log-to-vec .

# Run container
docker run -it --rm -v $(pwd)/data:/app/data log-to-vec
```

**Why Docker?** It ensures everyone uses the same OS + dependencies, which is critical for reproducible experiments and avoiding "it works on my machine" issues.

#### Option 3: Apptainer (for HPC clusters)

```bash
# Build the Apptainer container
apptainer build log-to-vec.sif apptainer.def

# Run with GPU support
apptainer exec --nv log-to-vec.sif python examples/train_toy_example.py
```

**Why Apptainer?** Many HPC clusters disallow Docker for security reasons. Apptainer runs containers without root privileges and works well on shared infrastructure.

### Generate Toy Data

```bash
# Generate a single log file
python examples/toy_log_generator.py --num-events 10000

# Generate multiple scenarios
python examples/toy_log_generator.py --scenarios
```

### Preprocess Logs to Numerical Features

Transform logs into numerical feature vectors:

```bash
# Basic preprocessing
python examples/preprocess_logs.py --input data/toy_logs.csv

# Create sequences with sliding window
python examples/preprocess_logs.py \
    --input data/toy_logs.csv \
    --sequence-length 10 \
    --stride 5 \
    --output data/processed_features.npz

# Visualize features
python examples/visualize_features.py --input data/processed_features.npz
```

See [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) for detailed documentation.

### Train Your First Model

```bash
# Train LSTM autoencoder on toy data
python examples/train_toy_example.py --config configs/toy_example.yaml

# Or train a simple classifier on preprocessed features
python examples/train_classifier.py --input data/toy_logs.csv
```

## Canonical Training Entrypoints

Use these scripts as the primary entrypoints in the unified branch:

- Reconstruction baseline: [examples/train_toy_example.py](examples/train_toy_example.py)
- Contrastive baseline (config-driven): [examples/train_contrastive_toy.py](examples/train_contrastive_toy.py)
- Version2 hybrid baseline: [examples/fsss/train_tcn_hybrid.py](examples/fsss/train_tcn_hybrid.py)

Compatibility script retained for older split-based workflows:

- [examples/train_contrastive.py](examples/train_contrastive.py)

## How to Add a New Model

1. **Create a model file** in [src/log_to_vec/models](src/log_to_vec/models) (e.g., `my_model.py`).
2. **Inherit** from `BaseEmbeddingModel` in [src/log_to_vec/models/base.py](src/log_to_vec/models/base.py).
3. **Register** the model in [examples/train_toy_example.py](examples/train_toy_example.py) so it can be selected via config.
4. **Add config** in [configs](configs) (e.g., `my_model.yaml`).

See the walkthrough in [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

## Where to Add New Loss Functions

- If the loss is tied to a model, implement it inside your model's `forward()` in [src/log_to_vec/models](src/log_to_vec/models).
- If you want a reusable loss, add a helper in [src/log_to_vec/training](src/log_to_vec/training) and call it from the training loop.
- Add tests in [tests](tests) for any new loss logic.

## Hyperparameter Search / Experiment Scripts

- Put sweep scripts in [experiments](experiments).
- Start by cloning the example pattern in [examples/train_toy_example.py](examples/train_toy_example.py), then loop over config variants.
- For clusters, run your scripts with Apptainer:

```bash
apptainer exec --nv log-to-vec.sif \
  python experiments/run_benchmark.py
```

## Where to Test Changes

- Unit tests live in [tests](tests). Add a new test file for your feature or loss.
- Use `pytest` to run tests:

```bash
pytest tests/
```

## Project Structure

```
log-to-vec/
├── README.md                      # This file
├── STUDENT_GUIDE.md              # Detailed guide for students
├── PREPROCESSING_GUIDE.md        # Guide for log preprocessing
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Dockerfile                    # Docker configuration
├── apptainer.def                 # Apptainer definition
├── configs/                      # Configuration files
│   └── toy_example.yaml
├── src/
│   └── log_to_vec/
│       ├── data/                 # Data handling
│       │   ├── log_parser.py    # Log parsing and tokenization
│       │   ├── preprocessor.py  # Numerical feature extraction
│       │   └── dataset.py       # PyTorch datasets
│       ├── models/               # Model implementations
│       │   ├── base.py          # Base classes
│       │   ├── autoencoder.py   # Autoencoder models
│       │   ├── transformer.py   # Transformer models (TODO)
│       │   └── contrastive.py   # Contrastive models (TODO)
│       ├── training/             # Training utilities
│       └── evaluation/           # Evaluation metrics
│           └── metrics.py
├── examples/                     # Example scripts
│   ├── toy_log_generator.py     # Generate synthetic logs
│   ├── preprocess_logs.py       # Preprocess to numerical features
│   ├── visualize_features.py    # Visualize preprocessed features
│   ├── train_classifier.py      # Train simple classifier
│   └── train_toy_example.py     # Train on toy data
├── experiments/                  # Experiment scripts
├── tests/                        # Unit tests
└── data/                        # Data directory (gitignored)
```

## Configuration

All experiments are configured via YAML files in the `configs/` directory. Key configuration sections:

- **data**: Dataset parameters (sequence length, splits, etc.)
- **model**: Model architecture and hyperparameters
- **training**: Training settings (batch size, learning rate, epochs)
- **evaluation**: Evaluation metrics and parameters
- **logging**: Logging and checkpointing settings
- **mode_change**: Optional change-point and segment clustering baseline settings

## Structured Experiment Tracking

Use the registry files under [experiments/registry](experiments/registry) to log each run in a machine-readable format.

- Schema: [experiments/registry/schema.json](experiments/registry/schema.json)
- Metadata template: [experiments/registry/templates/run_metadata.template.json](experiments/registry/templates/run_metadata.template.json)
- Run artifacts directory: [experiments/registry/runs/.gitkeep](experiments/registry/runs/.gitkeep)

Human-readable logs and merge decisions:

- Experiment log: [docs/experiment_log.md](docs/experiment_log.md)
- Branch merge ledger: [docs/merge_ledger.md](docs/merge_ledger.md)
- Distilled findings: [docs/research_findings.md](docs/research_findings.md)

## Alvis Cluster Helpers

Use helper scripts to standardize experiment execution on alvis1:

```bash
# Submit a SLURM job
scripts/alvis_submit.sh scripts/slurm/smoke_train.slurm

# Check job status
scripts/alvis_status.sh

# Collect artifacts from alvis1
scripts/alvis_collect.sh /path/on/alvis/results ./results
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black src/ examples/

# Check style
flake8 src/ examples/

# Sort imports
isort src/ examples/
```

## Contributing

This is a research project. Students should:

1. Fork the repository
2. Create a feature branch
3. Implement your method
4. Add tests and documentation
5. Submit a pull request

See [STUDENT_GUIDE.md](STUDENT_GUIDE.md) for detailed instructions.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{log_to_vec,
  title = {Log-to-Vec: Benchmarking AI Models for Log Embeddings},
  author = {Your Team},
  year = {2026},
  url = {https://github.com/yourusername/log-to-vec}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

This project is developed as part of a master's thesis project on log embedding methods.

## Contact

For questions and support, please open an issue on GitHub or contact the project maintainers.
