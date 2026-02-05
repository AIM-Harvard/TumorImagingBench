# TumorImagingBench Tutorials

Comprehensive guides for using and extending the TumorImagingBench framework.

## Quick Navigation

| Tutorial | Format | Topic | Time |
|----------|--------|-------|------|
| [00_getting_started.ipynb](./00_getting_started.ipynb) | Jupyter Notebook | Framework overview, check available models, quick examples | 10 min |
| [01_model_integration.ipynb](./01_model_integration.ipynb) | Jupyter Notebook | Add new foundation models to the framework | 20 min |
| [FEATURE_EXTRACTOR_GUIDE.md](./FEATURE_EXTRACTOR_GUIDE.md) | Markdown | Add new datasets and create feature extractors | 15 min |

## What You'll Learn

### Getting Started (Notebook)
- Framework components and architecture
- List and load available models
- Extract features from a single model
- Register custom models at runtime

### Model Integration (Notebook)
- Understand the BaseModel abstract class
- Implement a custom model extractor
- Input/output specifications
- Best practices and GPU optimization
- Testing your model implementation

### Feature Extraction Guide (Markdown)
- Prepare datasets in CSV format
- Create dataset-specific extractors
- Extract features for all models
- Save and load feature files
- Troubleshooting and best practices

## Framework Architecture

```
TumorImagingBench/
├── src/tumorimagingbench/
│   ├── models/                    # Foundation model extractors
│   │   ├── base.py               # BaseModel abstract class
│   │   ├── ct_fm.py       # Example implementation
│   │   └── [10+ models]          # FMCIB, VISTA3D, etc.
│   └── evaluation/                # Feature extraction pipeline
│       ├── base_feature_extractor.py
│       ├── dummy_dataset_feature_extractor.py
│       └── [dataset extractors]
└── tutorials/                     # Documentation (you are here)
```

## Typical Workflows

### Workflow 1: Extract Features with Existing Models
```bash
cd src/tumorimagingbench/evaluation
python nsclc_radiomics_feature_extractor.py \
  --output features/nsclc_radiomics.pkl \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --test-csv data/test.csv
```

### Workflow 2: Add a New Model
1. Open [01_model_integration.ipynb](./01_model_integration.ipynb)
2. Implement your model inheriting from `BaseModel`
3. Register it with `register_extractor()`
4. Use it in feature extraction pipeline

### Workflow 3: Add a New Dataset
1. Read [02_feature_extractor_guide.md](./02_feature_extractor_guide.md)
2. Prepare CSV files with required columns
3. Create `my_dataset_feature_extractor.py`
4. Run to extract features for all models

## Key Concepts

### BaseModel Interface
All models implement three methods:
- `load(weights_path)` - Load pre-trained weights
- `preprocess(x)` - Convert input dict to tensor
- `forward(x)` - Extract features, return numpy array

### Dataset Format
Each dataset needs:
- CSV files (train, val, test) with columns: `image_path`, `coordX`, `coordY`, `coordZ`, optional `label`
- NIFTI format images (.nii.gz)
- Physical coordinates in millimeters

### Feature Extraction Pipeline
```
CSV → get_split_data() → preprocess_row() → model.preprocess()
→ model.forward() → Features → save_features()
```

## Available Models

The framework includes 10+ foundation models:
- **CT-FM**: CT Foundation Model
- **FMCIB**: Foundation Model for Cancer Imaging
- **VISTA3D**: Medical Image Segmentation foundation model
- **SUPREM**: Segmentation and Representation Learning
- **PASTA**: Patch Aggregated Spatial Transformer
- **Merlin**: Medical Image Embedding
- **Voco**: Vision-based Contrastive Learning
- **ModelsGen**: Generative Medical Models
- And more...

Check available models:
```python
from tumorimagingbench.models import get_available_extractors
print(get_available_extractors())
```

## Common Tasks

### Check Available Models
```python
from tumorimagingbench.models import get_available_extractors
models = get_available_extractors()
print(f"Available: {models}")
```

### Load and Test a Model
```python
from tumorimagingbench.models import get_extractor
import torch

Model = get_extractor('CTFMExtractor')
model = Model()
model.load()
model = model.to('cuda')

# Extract features
dummy = torch.randn(1, 1, 48, 48, 48, device='cuda')
features = model.forward(dummy)
print(f"Features shape: {features.shape}")
```

### Register a Custom Model
```python
from tumorimagingbench.models import register_extractor

register_extractor('MyModelName', MyModelClass)
```

## Requirements

- Python >= 3.10, < 3.12
- PyTorch with CUDA support
- MONAI for medical imaging operations
- pandas, numpy for data handling

Install dependencies:
```bash
uv sync
```

## Troubleshooting

**Models not loading?** → Check [00_getting_started.ipynb](./00_getting_started.ipynb) for dependency issues

**Feature extraction fails?** → Check [FEATURE_EXTRACTOR_GUIDE.md](./FEATURE_EXTRACTOR_GUIDE.md) troubleshooting section

**Want to add a new model?** → Follow [01_model_integration.ipynb](./01_model_integration.ipynb)

**Want to add a new dataset?** → Follow [FEATURE_EXTRACTOR_GUIDE.md](./FEATURE_EXTRACTOR_GUIDE.md)

## Next Steps

1. Start with [00_getting_started.ipynb](./00_getting_started.ipynb) for overview
2. Go to [01_model_integration.ipynb](./01_model_integration.ipynb) to learn about models
3. Read [FEATURE_EXTRACTOR_GUIDE.md](./FEATURE_EXTRACTOR_GUIDE.md) for datasets
4. Check existing extractors in `src/tumorimagingbench/evaluation/` for reference implementations

## Reference Documentation

- **Base Model**: `src/tumorimagingbench/models/base.py`
- **Feature Extraction**: `src/tumorimagingbench/evaluation/base_feature_extractor.py`
- **Example Models**: `src/tumorimagingbench/models/ct_fm.py`
- **Example Extractors**: `src/tumorimagingbench/evaluation/dummy_dataset_feature_extractor.py`
