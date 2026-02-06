# Student Guide: Log-to-Vec Project

Welcome! This guide will help you get started with the Log-to-Vec benchmarking project. This project is designed for master's students working on embedding methods for time-stamped log data.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Project](#understanding-the-project)
3. [Running the Toy Example](#running-the-toy-example)
4. [Implementing Your Own Model](#implementing-your-own-model)
5. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
6. [Working on the Cluster](#working-on-the-cluster)
7. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### Prerequisites

- Python 3.8+ (recommended: 3.10)
- CUDA-capable GPU (optional but recommended for training)
- Basic understanding of PyTorch
- Familiarity with time series and sequence modeling

### Environment Setup

#### Using Docker with VS Code (Recommended)

**Docker ensures everyone uses identical dependencies and OS configuration**, which eliminates "works on my machine" issues and makes experiments reproducible.

**Windows Users**: Install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) first. Docker Desktop for Windows requires WSL 2, and you'll get better performance developing inside WSL rather than directly on Windows.

**Setup Steps**:

1. **Install Prerequisites**:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Engine)
   - [Visual Studio Code](https://code.visualstudio.com/)
   - VS Code extension: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - (Windows only) [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)

2. **Clone the repository**:
   ```bash
   # On Windows, clone inside WSL:
   # Open WSL terminal, then:
   git clone <repository-url>
   cd log-to-vec
   ```

3. **Open in VS Code Dev Container**:
   
   **Windows/WSL users**:
   ```bash
   # From WSL terminal, in the log-to-vec folder:
   code .
   ```
   This opens VS Code with WSL integration already configured.
   
   **All users**:
   - When VS Code opens, you'll see a prompt: "Reopen in Container"
   - Click **Reopen in Container**
   - Or manually: Press `F1` → type "Dev Containers: Reopen in Container"
   
   VS Code will build the Docker image and open a terminal inside the container. All your editing and terminal commands now run in the containerized environment.

4. **Verify setup** (inside the container):
   ```bash
   python -c "from src.log_to_vec.data import LogPreprocessor; print('Success!')"
   ```

**Why this workflow?**
- Same environment for everyone (no dependency conflicts)
- GPU support works out of the box
- Easy to replicate on HPC clusters (using Apptainer)
- No risk of breaking your host system

#### Alternative: Docker Command Line

If you prefer not to use VS Code Dev Containers:

```bash
# Build the image
docker build -t log-to-vec .

# Run interactively with GPU support
docker run -it --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints log-to-vec

# Inside the container, run commands
python examples/toy_log_generator.py --scenarios
```

#### Local Development (Without Docker)

If you cannot use Docker, set up a local Python environment:

1. **Clone and navigate to the repository**:
   ```bash
   cd log-to-vec
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac/WSL
   # or
   venv\Scripts\activate  # Windows (not recommended, use WSL instead)
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   python -c "from src.log_to_vec.data import LogPreprocessor; print('Success!')"
   ```

**Note**: Local setup may have dependency conflicts or OS-specific issues. Docker/Dev Containers are strongly recommended.

## Understanding the Project

### Project Goals

The main objectives are to:

1. **Study existing methods** for log embeddings (autoencoders, Transformers, contrastive learning)
2. **Implement and compare** multiple approaches
3. **Define evaluation metrics** for unsupervised embeddings
4. **Benchmark systematically** across different architectures and hyperparameters

### Key Concepts

#### Log Parsing
- **Tokenization**: Converting log events into discrete tokens
- **Vocabulary**: Mapping between event types and integer IDs
- **Temporal Features**: Capturing time deltas between events

#### Embedding Methods
- **Reconstruction-based**: Train models to reconstruct input sequences (autoencoders)
- **Contrastive**: Learn representations by contrasting positive/negative pairs
- **Masked Modeling**: Predict masked tokens (BERT-style)

#### Evaluation
- **Reconstruction Error**: How well can we reconstruct the original sequence?
- **Similarity Search**: Can we find similar log sequences?
- **Clustering**: Do embeddings group similar behaviors?
- **Anomaly Detection**: Can we identify unusual patterns?

## Running the Toy Example

### Step 1: Generate Synthetic Logs

The toy log generator creates realistic PLC-like system logs:

```bash
# Generate a single scenario
python examples/toy_log_generator.py --num-events 10000 --output-dir data

# Generate multiple scenarios (normal, high anomaly, long sequence)
python examples/toy_log_generator.py --scenarios --output-dir data

# Customize generation
python examples/toy_log_generator.py \
    --num-events 20000 \
    --anomaly-rate 0.1 \
    --seed 123 \
    --output-dir data/custom
```

**What's generated?**
- Time-stamped events from a simulated PLC system
- Multiple event types (sensor reads, actuator commands, alarms, etc.)
- Realistic temporal patterns
- Occasional anomalies (error bursts, state transitions)

**Output format** (CSV):
```
timestamp,event_type,component,severity,message,data
2026-02-03 10:00:00,SYSTEM_START,MainController,INFO,SYSTEM_START in RUNNING state,
2026-02-03 10:00:01.234,SENSOR_READ,TemperatureSensor,INFO,SENSOR_READ in RUNNING state,"{'temperature': 25.3, 'pressure': 5.2}"
...
```

### Step 2: Train on Toy Data

Train an LSTM autoencoder on the generated logs:

```bash
# Using default config
python examples/train_toy_example.py

# Using custom config
python examples/train_toy_example.py --config configs/toy_example.yaml

# Override data file
python examples/train_toy_example.py --data-file data/toy_logs_high_anomaly.csv
```

**What happens during training?**

1. **Data Loading**: Reads CSV log file
2. **Parsing**: Builds vocabulary and tokenizes events
3. **Feature Extraction**: Extracts events, timestamps, severity, etc.
4. **Dataset Creation**: Creates sliding windows over log sequences
5. **Training**: Trains autoencoder to reconstruct sequences
6. **Validation**: Evaluates reconstruction and embedding quality
7. **Checkpointing**: Saves best model to `checkpoints/`

**Expected output**:
```
Epoch 1/50
Train Loss: 2.3456, Train Accuracy: 0.3245
Val Loss: 2.1234, Val Accuracy: 0.3567

Epoch 10/50
Evaluating embeddings...
  similarity/avg_neighbor_distance: 0.4523
  similarity/mean_similarity: 0.6234
  clustering/silhouette_score: 0.3456
...

Saved best model to checkpoints/toy_example/best_model.pt
```

### Step 3: Understand the Configuration

Open `configs/toy_example.yaml` and explore the settings:

```yaml
data:
  sequence_length: 100    # How many events per sequence
  stride: 50              # Overlap between sequences
  
model:
  type: "autoencoder"     # Model type
  embedding_dim: 128      # Size of embeddings
  hidden_dim: 256         # Hidden layer size
  
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

**Exercise**: Try modifying these parameters:
- Increase `embedding_dim` to 256 - what happens to training time?
- Change `sequence_length` to 50 or 200 - how does it affect accuracy?
- Try `model.type: "transformer"` instead of `"autoencoder"`

## Implementing Your Own Model

### Step 1: Create a New Model Class

Create a new file `src/log_to_vec/models/my_model.py`:

```python
"""My custom embedding model."""

import torch
import torch.nn as nn
from .base import BaseEmbeddingModel

class MyCustomModel(BaseEmbeddingModel):
    """Your custom model description."""
    
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(vocab_size, embedding_dim)
        
        # Define your architecture here
        self.encoder = nn.Sequential(
            # Your layers...
        )
    
    def encode(self, batch):
        """Encode sequences to embeddings."""
        events = batch["events"]
        embedded = self.token_embedding(events)
        
        # Your encoding logic
        embeddings = self.encoder(embedded)
        
        return embeddings
    
    def forward(self, batch):
        """Forward pass."""
        embeddings = self.encode(batch)
        
        # Your forward logic (e.g., reconstruction, contrastive loss)
        
        return {
            "embeddings": embeddings,
            # other outputs...
        }
```

### Step 2: Register Your Model

Add your model to the training script. In `examples/train_toy_example.py`, add:

```python
from log_to_vec.models.my_model import MyCustomModel

# In the model creation section:
elif config["model"]["type"] == "my_custom":
    model = MyCustomModel(
        vocab_size=vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        # your parameters...
    )
```

### Step 3: Create a Configuration

Create `configs/my_model.yaml`:

```yaml
data:
  log_file: "data/toy_logs.csv"
  sequence_length: 100
  stride: 50

model:
  type: "my_custom"
  embedding_dim: 128
  # your hyperparameters...

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

### Step 4: Train and Evaluate

```bash
python examples/train_toy_example.py --config configs/my_model.yaml
```

## Evaluation and Benchmarking

### Understanding Metrics

#### Reconstruction Metrics
- **Accuracy**: How many tokens are correctly reconstructed?
- **Cross-Entropy Loss**: Probability of correct reconstruction

#### Embedding Quality Metrics
- **Silhouette Score**: Measures cluster quality (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Average Neighbor Distance**: Cosine distance to k nearest neighbors

#### Retrieval Metrics
- **Precision@K**: Fraction of relevant items in top K results
- **Recall@K**: Fraction of relevant items retrieved

### Creating Custom Evaluation

Add your own metrics to `src/log_to_vec/evaluation/metrics.py`:

```python
def my_custom_metric(embeddings, labels=None):
    """Your custom evaluation metric."""
    # Implement your metric
    score = ...
    return score
```

### Running Systematic Benchmarks

Create a benchmark script `experiments/run_benchmark.py`:

```python
import itertools
from pathlib import Path

# Define hyperparameter grid
configs = {
    "embedding_dim": [64, 128, 256],
    "hidden_dim": [128, 256, 512],
    "learning_rate": [0.001, 0.0001],
}

# Run experiments
for params in itertools.product(*configs.values()):
    config = dict(zip(configs.keys(), params))
    # Train model with these parameters
    # Save results
```

## Working on the Cluster

### Building Apptainer Container

On your local machine or the cluster login node:

```bash
apptainer build log-to-vec.sif apptainer.def
```

### Submitting Jobs

Create a SLURM job script `job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=log2vec
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j_output.txt
#SBATCH --error=logs/%j_error.txt

# Load modules (if needed)
module load apptainer

# Run training
apptainer exec --nv log-to-vec.sif \
    python examples/train_toy_example.py \
    --config configs/toy_example.yaml
```

Submit the job:

```bash
sbatch job.sh
```

### Interactive Session

For debugging, request an interactive GPU session:

```bash
srun --pty --gres=gpu:1 --mem=16G --time=2:00:00 bash

# Once allocated
apptainer shell --nv log-to-vec.sif
python examples/train_toy_example.py
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/JOBID_output.txt

# Cancel job
scancel JOBID
```

## Tips and Best Practices

### Development Workflow

1. **Start small**: Begin with the toy example to understand the pipeline
2. **Iterate quickly**: Test on small datasets (1000 events) before scaling up
3. **Version control**: Commit frequently, use meaningful commit messages
4. **Document**: Add docstrings to your functions and classes
5. **Experiment tracking**: Use TensorBoard or Weights & Biases

### Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Print tensor shapes
print(f"Embeddings shape: {embeddings.shape}")

# Check for NaN
assert not torch.isnan(loss).any(), "Loss is NaN!"

# Visualize embeddings with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.savefig("embeddings_tsne.png")
```

### Performance Tips

1. **Use GPU**: Always check `device` is set to 'cuda' when available
2. **Batch processing**: Use larger batch sizes if memory allows
3. **DataLoader workers**: Set `num_workers > 0` for faster data loading
4. **Mixed precision**: Use `torch.cuda.amp` for faster training
5. **Profiling**: Use PyTorch profiler to identify bottlenecks

### Common Issues

**Issue**: `CUDA out of memory`
- **Solution**: Reduce batch size, sequence length, or model size

**Issue**: Model doesn't converge
- **Solution**: Check learning rate (try 0.001, 0.0001), verify loss computation

**Issue**: Embeddings are all similar
- **Solution**: Check if model is learning (loss decreasing?), try different architecture

**Issue**: Import errors
- **Solution**: Make sure you installed with `pip install -e .`

## Project Milestones

### Week 1-2: Setup and Understanding
- [ ] Set up environment
- [ ] Run toy example successfully
- [ ] Understand codebase structure
- [ ] Read related papers on log embeddings

### Week 3-4: Implementation
- [ ] Implement your first model variant
- [ ] Train on toy data
- [ ] Evaluate and compare with baseline

### Week 5-6: Experimentation
- [ ] Implement 2-3 different approaches
- [ ] Run systematic hyperparameter search
- [ ] Collect results and metrics

### Week 7-8: Benchmarking
- [ ] Compare all methods on standardized tasks
- [ ] Analyze strengths/weaknesses
- [ ] Generate visualizations

### Week 9-10: Real Data (Optional)
- [ ] Test on real log datasets
- [ ] Evaluate generalization
- [ ] Fine-tune best models

### Week 11-12: Documentation
- [ ] Write technical report
- [ ] Prepare presentation
- [ ] Clean up code and documentation

## Resources

### Papers to Read
- **Autoencoders**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- **Transformers**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Contrastive Learning**: "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)
- **Log Analysis**: "Deep Learning for Log Analysis" (survey papers)

### PyTorch Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Tools
- **TensorBoard**: Visualize training curves
- **Weights & Biases**: Experiment tracking
- **Jupyter**: Interactive development
- **VS Code**: Recommended IDE with Python extension

## Getting Help

1. **Check documentation**: README.md, docstrings, comments
2. **Search issues**: Look for similar problems in GitHub issues
3. **Ask questions**: Open a GitHub issue or contact your supervisor
4. **Debug systematically**: Isolate the problem, check inputs/outputs
5. **Pair programming**: Work with other students when stuck

## Final Notes

This project is your opportunity to:
- Learn state-of-the-art embedding methods
- Gain experience with PyTorch and deep learning
- Develop research and experimental skills
- Contribute to an important problem in log analysis

**Remember**: 
- Start simple, iterate quickly
- Document your work as you go
- Don't hesitate to ask for help
- Have fun experimenting!

Good luck! 🚀
