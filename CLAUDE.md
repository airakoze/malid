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

## Applying Mal-ID to HIV: time-to-rebound

Mal-ID is classification-first, but it can support time-to-event (e.g., time to viral rebound after ATI) via a two-stage approach: use Mal-ID to produce immune-repertoire features, then fit a survival model on those features.

### Data you need
- **Sequences + metadata** for HIV participants (BCR and/or TCR), ideally at a consistent pre-ATI timepoint.
- Extend `metadata/hiv_cohort.specimens.tsv` (or a separate table you merge in ETL) with survival fields:
  - `participant_label`, `specimen_label`, `disease`, `disease_subtype` (use existing conventions)
  - `specimen_time_point` (e.g., "Pre-ATI" or a day string like `0 days`)
  - `time_to_rebound_days` (numeric, duration until rebound or censor)
  - `rebounded` (1 if observed rebound, 0 if right-censored)
  - Optionally `ati_date`, `rebound_date` (for provenance)
- Optional: set `hiv_run_filter=True` for HIV participants (ETL respects this to include only allowed runs).

### Ingest/update metadata
- Place your updated table in `metadata/` and add a merge in `notebooks_src/assemble_etl_metadata.py` (see the existing HIV block). Regenerate the combined table:

```bash
./run_notebooks.sh notebooks/assemble_etl_metadata.ipynb
```

This writes `metadata/generated_combined_specimen_metadata.tsv` used downstream.

### Generate embeddings for the cohort
Follow the external-cohort path (fold `-1`) as in the README:

```bash
python scripts/run_embedding.py --fold_id -1 --external-cohort 2>&1 | tee data/logs/external_validation_cohort_embedding.log
python scripts/scale_embedding_anndatas.py --fold_id -1 --external-cohort 2>&1 | tee data/logs/external_validation_cohort_scaling.log
```

### Featurize with Mal-ID metamodels
- Load metamodels and featurize to obtain specimen-level feature matrices (see `notebooks/evaluate_external_cohorts.ipynb` for a working pattern). You’ll get:
  - `X`: features per specimen
  - `metadata`: includes `participant_label`, `specimen_label`, and any merged columns (`time_to_rebound_days`, `rebounded`, `specimen_time_point`, etc.)
- Reduce to one row per participant for survival (recommended: the pre-ATI specimen; if multiple, choose earliest or average features across pre-ATI specimens).

### Fit survival models (outside Mal-ID core)
Use the featurized matrix as covariates in a survival model:
- **Cox proportional hazards** (e.g., with `lifelines`): models hazard as exp(linear combination of Mal-ID features)
- **Random Survival Forest** (e.g., `scikit-survival`) for non-linear effects
- **XGBoost survival** (Cox or AFT objectives) if preferred

Typical inputs: `duration_col = time_to_rebound_days`, `event_col = rebounded`, covariates = Mal-ID features `X` (optionally plus demographics from `metadata`). Evaluate with concordance index and time-dependent metrics.

### Alternative: discretize time and use classification
If you’d rather stay within Mal-ID’s native classification workflow:
- Bin time-to-rebound into categories (e.g., Early vs Late vs No rebound by a study-defined threshold), add that label to metadata, and train a classifier using the existing pipeline (see `malid/datamodels.py` for examples of target definitions).
- This loses time resolution but requires no external survival tooling.

### Notes
- `helpers.get_all_specimen_info()` computes `specimen_time_point_days` for day-like strings; keep timepoint strings consistent to ease filtering.
- Ensure group-aware splits by participant when doing any CV on survival models (no leakage across the same participant).
- Control for demographics if desired: the metamodel supports adding or regressing out covariates; see `train/train_metamodel.py` and `trained_model_wrappers/blending_metamodel.py` for hooks.