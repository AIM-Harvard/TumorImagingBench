# TumorImagingBench

TumorImagingBench is a framework for extracting foundation model embeddings from medical images and benchmarking them across radiomics datasets.

**Overview**
- Unified interface for multiple foundation model extractors.
- Dataset-specific feature extraction pipelines.
- Analysis workflows in notebooks for performance, robustness, and stability.

**Repository Structure**
```
TumorImagingBench/
├── src/tumorimagingbench/     # Core package (models, evaluation)
├── scripts/                   # Utility scripts
├── tutorials/                 # Tutorials and guides
├── notebooks/                 # Analysis notebooks
├── data/                      # Datasets (ignored by git)
├── dist/                      # Large weights (ignored by git)
├── metrics/                   # Evaluation outputs
└── plots/                     # Figures and plots
```

**Installation**
```bash
uv sync
uv run python -m pip install -e .
```

Python requirement: `>=3.10,<3.12`.

**Quickstart**
List available extractors:
```python
from tumorimagingbench.models import get_available_extractors

print(get_available_extractors())
```

Load a model and initialize weights:
```python
from tumorimagingbench.models import get_extractor

Model = get_extractor("VISTA3DExtractor")
model = Model()
model.load()
```

**Feature Extraction**
Example using the LUNA16 extractor:
```bash
uv run python src/tumorimagingbench/evaluation/luna_feature_extractor.py \
  --output features/luna.pkl \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --test-csv /path/to/test.csv
```

Notes:
- Dataset CSVs should include `image_path`, `coordX`, `coordY`, `coordZ` (and optional labels).
- Many extractors ship with absolute default paths; override them via flags.
- Feature extraction expects a CUDA-capable GPU.

**Supported Models**
- `CTClipVitExtractor`
- `CTFMExtractor`
- `FMCIBExtractor`
- `MedImageInsightExtractor`
- `MerlinExtractor`
- `ModelsGenExtractor`
- `PASTAExtractor`
- `SUPREMExtractor`
- `VISTA3DExtractor`
- `VocoExtractor`

**Supported Datasets**
- LUNA16
- DLCS (Duke Lung Cancer Screening)
- NSCLC Radiomics
- NSCLC Radiogenomics
- C4C-KiTS
- Colorectal Liver Metastases
- LNDb
- RIDER (test-retest stability)

**Tutorials**
- See `tutorials/README.md` for guided notebooks and dataset/model integration walkthroughs.

**Evaluation**
- Example modelling workflow: `notebooks/modelling/luna_modelling.ipynb` (LUNA16 evaluation notebook).
- Loads extracted features from `data/features/luna.pkl` and evaluates per-model performance.
- Baselines: k-NN probing with AUC and 95% CI; linear probing (logistic regression); few-shot (1/5/10-shot).
- Visual outputs saved to `plots/` (e.g., `luna_auc.png`, `luna_knn_overlap.png`, `luna_few_shot.png`, `luna_evaluation_protocols.png`).
- Ensemble methods: alignment-weighted k-NN and stacking meta-learner; weight and comparison plots (e.g., `luna_alignment_weights.png`, `luna_stacking_weights.png`, `luna_ensemble_comparison.png`, `luna_ensemble_vs_individual.png`).
- Aggregates results into `overall_results.csv`.

**Contributing**
- Follow the existing code style and update docs with changes.
- Add targeted tests for new functionality.

**Citation**
```bibtex
@article{TumorImagingBench,
  title={Foundation model embeddings for quantitative tumor imaging biomarkers},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  publisher={}
}
```

**License**
MIT. See `LICENSE`.
