# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mal-ID is a machine learning-based disease classification system using B cell receptor (BCR) and T cell receptor (TCR) sequencing data. This is a research implementation accompanying the paper "Disease diagnostics using machine learning of immune receptors".

## Essential Commands

### Development
```bash
make lint           # Run pre-commit hooks on all files
make lint-staged    # Run pre-commit hooks on staged files
make test           # Run pytest
make regen-jupytext # Regenerate Python scripts from notebooks
make clean          # Remove build artifacts
```

### Testing
```bash
pytest                    # Run all tests
pytest --exitfirst       # Stop on first failure
pytest -m "not slow"     # Skip slow tests
pytest -m gpu            # Run only GPU tests (requires GPU)
pytest tests/test_specific.py::TestClass::test_method  # Run single test
```

### Notebook Execution
```bash
./run_notebooks.sh notebook1.ipynb notebook2.ipynb  # Execute notebooks
```

## High-Level Architecture

### Core ML Pipeline Structure
The codebase follows a modular pipeline architecture:

1. **Data Ingestion & ETL** (`malid/etl.py`, `malid/io.py`): Handles loading BCR/TCR sequence data and metadata from various formats.

2. **Preprocessing** (`preprocessing/`): Raw data processing, IgBlast alignment, sequence clustering, and quality control.

3. **Feature Engineering & Embeddings** (`malid/embedders/`): Multiple protein language model implementations (ablang, biotransformers, unirep) generate sequence embeddings.

4. **Model Training** (`malid/train/`): Different classifier implementations (sequence-based, convergent cluster-based, repertoire stats-based) with cross-validation support.

5. **Model Wrappers** (`malid/trained_model_wrappers/`): Unified interfaces for trained models enabling consistent prediction and evaluation.

### Key Design Patterns

**Notebook-Script Pairing**: All notebooks in `notebooks/` have corresponding Python scripts in `notebooks_src/` via jupytext. Always edit the `.py` files and regenerate notebooks with `make regen-jupytext`.

**Configuration Hierarchy**: 
- Global config in `malid/config.py` controls feature flags and paths
- Environment variables (e.g., `MALID_DATASET_VERSION`) override defaults
- Individual modules have their own configuration patterns

**GPU/CPU Abstraction**: Code automatically detects and uses GPU acceleration when available via RAPIDS. Falls back to CPU implementations seamlessly.

**Cross-Validation Framework**: Multiple CV strategies (e.g., LeaveOneOut, StratifiedKFold) are implemented consistently across all model types.

### Data Flow

1. Raw sequence data → IgBlast processing → Filtered sequences
2. Sequences → Language model embeddings → Feature vectors
3. Feature vectors + metadata → Model training with CV
4. Trained models → Evaluation on test sets → Performance metrics

### Critical Integration Points

- **`malid/datamodels.py`**: Central data structures (SequenceDataset, SampleWeights, TargetObsColumnEnum)
- **`malid/helpers.py`**: Utility functions used throughout the codebase
- **`malid/external/`**: External utilities and specialized functions
- **`scripts/`**: Entry points for training and evaluation workflows

## Testing Philosophy

Tests use pytest with GPU/CPU markers. Test data snapshots in `tests/snapshot/` ensure reproducibility. Always run tests before committing changes to core modules.

## Environment Considerations

- GPU environment preferred for performance (RAPIDS, CUDA 11.8)
- CPU-only fallback available
- Python 3.9 required
- JAX memory management via `XLA_PYTHON_CLIENT_PREALLOCATE=false` for GPU