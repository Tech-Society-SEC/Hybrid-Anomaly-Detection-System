# Hybrid Anomaly Detection System

A modular, extensible system for detecting anomalies using a hybrid approach that combines statistical, machine-learning, and (optionally) deep-learning techniques. Designed for research and production experimentation with tabular and time-series datasets.


## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Visualizations and Notebooks](#visualizations-and-notebooks)
- [Experiment Tracking](#experiment-tracking)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview
Hybrid-Anomaly-Detection-System (HADS) is intended to provide a unified framework for:
- Preprocessing and feature engineering for anomaly detection datasets (time-series and tabular).
- Running multiple anomaly detection algorithms (statistical baselines, classic ML, and neural models).
- Training, evaluating, and comparing models with consistent metrics and visualizations.
- Exporting models and inference pipelines for deployment.

This repository focuses on reproducibility and experimentation: configuration-driven runs, clear separation between data, models, and evaluation, and (optionally) containerized execution.

## Key Features
- Data ingestion and preprocessing pipelines
- Modular model implementations (plug-in friendly)
- Training and evaluation scripts with standard metrics (precision, recall, F1, AUC)
- Inference pipeline for single-sample and batch predictions
- Support for experimenting with hybrid approaches (ensemble / stacked detectors)
- Jupyter notebooks for exploratory analysis and visualization
- (Optional) Docker support for reproducible environments

## Architecture
A typical project layout (adjust to the repository structure if different):
- data/              -> raw and processed datasets (not always checked in)
- src/
  - preprocessing/   -> scalers, feature builders, windowing
  - models/          -> model implementations and wrappers
  - trainers/        -> training loops and checkpointing
  - inference/       -> inference pipelines and utilities
  - evaluation/      -> metrics and evaluation scripts
  - utils/           -> logging, config parsing, helpers
- notebooks/         -> EDA and demo notebooks
- configs/           -> YAML/JSON configuration files for experiments
- requirements.txt   -> Python dependencies
- Dockerfile         -> optional container image

## Requirements
- Python 3.8+
- pip
- (Optional) GPU + CUDA for deep-learning models
- Recommended virtual environment: venv or conda

Python dependencies will typically be listed in requirements.txt or environment.yml. Example packages frequently used:
- numpy, pandas, scikit-learn
- scipy
- matplotlib, seaborn
- pytorch or tensorflow (only if neural networks are used)
- yaml, hydra/omegaconf (optional, for config management)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Tech-Society-SEC/Hybrid-Anomaly-Detection-System.git
   ```
   <br>
   
      ```bash cd Hybrid-Anomaly-Detection-System```

3. Create and activate a virtual environment:
   ```python -m venv venv```
   ```source venv/bin/activate```   # Linux / macOS
   ```venv\Scripts\activate```      # Windows

4. Install dependencies:
   ```pip install -r requirements.txt```

Optional (Docker):
- Build: docker build -t hads:latest .
- Run: docker run --rm -it -v $(pwd):/workspace hads:latest /bin/bash

## Quick Start
1. Prepare your dataset in data/raw or update the path in the config.
2. Preprocess data:
   python src/preprocessing/run_preprocessing.py --config configs/preprocess.yaml
3. Train a model:
   python src/train.py --config configs/train.yaml
4. Run evaluation:
   python src/evaluate.py --config configs/evaluate.yaml
5. Run inference on new data:
   python src/infer.py --config configs/infer.yaml

Adjust the above commands to match actual script filenames and config structure present in the repository.

## Configuration
The project uses configuration files (YAML/JSON) to control experiments:
- configs/preprocess.yaml
- configs/train.yaml
- configs/evaluate.yaml
Each config should include data paths, model hyperparameters, training settings, and output locations (models/, logs/, results/).

## Data
- Place raw datasets under data/raw/.
- Processed datasets should go to data/processed/.
- For time-series, expected format: timestamp column, feature columns, and an optional label column (0 = normal, 1 = anomaly).
- For tabular datasets, include a label column and a unique id if needed.

If you plan to use public datasets, add download and preprocessing steps in scripts or notebooks.

## Training
- Training scripts should:
  - Load the config
  - Prepare dataset splits (train/val/test)
  - Initialize model(s) and optimizer(s)
  - Log metrics and save checkpoints (models/)
- Use early stopping and model checkpointing for neural models.
- Save model artifacts and the exact config used for the run to reproduce results.

Example:
python src/train.py --config configs/train.yaml

## Inference
- Provide an inference script that accepts:
  - model checkpoint path
  - input data (single sample or batch)
  - optional threshold for decision making
- Output should include:
  - anomaly score(s)
  - binary anomaly predictions
  - any metadata required for downstream systems

Example:
python src/infer.py --model models/best_checkpoint.pth --input data/sample.csv

## Evaluation
- Common metrics:
  - Precision, Recall, F1-score
  - ROC-AUC, PR-AUC
  - Detection delay (for time-series)
- Include scripts to generate:
  - Per-class and aggregate metrics
  - Confusion matrices
  - Time-series plots with anomaly windows highlighted

Example:
python src/evaluate.py --config configs/evaluate.yaml

## Visualizations and Notebooks
- Keep notebooks for:
  - Exploratory data analysis
  - Demo inference runs
  - Comparing detectors visually
- Notebooks should reference saved model artifacts and processed data (avoid committing large data to the repo).

## Experiment Tracking
- Optional integrations:
  - MLflow
  - Weights & Biases
  - TensorBoard
- If used, store experiment runs, parameters, and artifact links in configs and README.

## Testing
- Include unit tests for critical preprocessing steps, metric computations, and model wrappers.
- Run tests with pytest:
  pytest tests/

## Development & Contribution
- Please open issues for bugs or feature requests.
- To contribute:
  1. Fork the repository
  2. Create a feature branch: git checkout -b feature/my-change
  3. Make changes and add tests
  4. Submit a pull request describing your changes

Follow repository coding standards and include clear tests where applicable.

## License
See the LICENSE file in this repository for license details. If no LICENSE is present, add an appropriate license (e.g., MIT, Apache-2.0) before redistributing.

## Contact
Maintainers:
- Tech-Society-SEC / project contributors :
- Karnala Santhan Kumar
- Hanshika Varthini R

For questions or help, open an issue or contact the repository maintainers via their GitHub profiles.

## Acknowledgements
- Inspired by standard anomaly detection research and open-source toolkits.


## Docker Instructions :


Files Created:
Dockerfile - Multi-stage build configuration
docker-compose.yml - Orchestration with volume mounts for hot-reload
.dockerignore - Optimize build context
docker-deploy.sh - Interactive deployment script (Linux/Mac)
docker-deploy.bat - Interactive deployment script (Windows)
DOCKER_DEPLOYMENT.md - Complete deployment guide
Updated README.md - Added Docker quick start


🚀 Usage:
Windows :
```
# Run interactive menu
.\docker-deploy.bat
```
Linux/Mac :
```
chmod +x docker-deploy.sh
./docker-deploy.sh
```
Manual Commands :
```
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

✨ Key Features:
✅ Cross-platform - Works identically on Windows, Linux, macOS
✅ Hot-reload - Code changes reflect immediately (no rebuild)
✅ Health checks - Automatic container monitoring
✅ Volume mounts - Persistent model files and data
✅ Interactive scripts - Easy deployment with menus
✅ Production-ready - Includes Gunicorn configuration

🔄 Development Workflow:
Clone repo on any device
Run docker-compose up -d
Edit code (auto-reloads)
Access at http://localhost:5000
No Python/TensorFlow installation needed! Docker handles everything.

