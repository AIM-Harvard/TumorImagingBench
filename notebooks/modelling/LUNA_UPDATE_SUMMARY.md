# LUNA Modelling Notebook - Advanced Methods Integration

## Summary of Changes

The `luna_modelling.ipynb` notebook has been updated to include three new advanced evaluation protocols alongside the existing KNN probing analysis.

---

## What Was Added

### 1. Updated Imports (Cell 0)
- Added `roc_auc_score` from sklearn.metrics
- Added plotly subplots for visualizations
- Added new methods:
  - `train_linear_probing_classifier`
  - `train_few_shot_classifier`
  - `build_knn_ensemble_classifier`
  - `predict_with_ensemble`

### 2. Linear Probing Section (Cells 8-9)

**Cell 8**: Markdown header and description
**Cell 9**: Code implementation

Evaluates foundation models using logistic regression on frozen features.

**What it does:**
- Trains linear classifier on each model's features
- Evaluates across 10 shuffle splits with stratification
- Computes mean AUC with 95% confidence intervals
- Provides complementary evaluation to KNN probing

**Key Output:**
```
linear_probing_results[model_name] = {"mean": auc, "ci95": (lower, upper)}
```

### 3. Comparison Analysis (Cells 10-11)

**Cell 10**: Markdown header
**Cell 11**: Code for grouped comparison visualization

**Visualizations created:**
- Grouped bar chart comparing KNN, Linear Probing, and 10-Shot learning
- Correlation statistics between methods
- Interpretation guide

**Key Metrics Computed:**
- Pearson correlation between KNN and Linear Probing (should be >0.8)
- Correlation between KNN and few-shot learning
- Performance differences highlighting data-efficiency

### 4. Few-Shot Learning Section (Cells 12-13)

**Cell 12**: Markdown header and explanation
**Cell 13**: Code implementation for 1, 5, and 10-shot evaluation

**What it does:**
- Trains k-NN classifiers with limited samples (1, 5, 10 per class)
- Optimizes k hyperparameter for each shot configuration
- Evaluates across 10 splits with stratification
- Assesses how well models generalize with minimal labeled data

**Key Output:**
```
few_shot_results[shots][model_name] = {"mean": auc, "ci95": (lower, upper)}
```

**Example shot configurations:**
- 1-shot: Extreme data scarcity (1 sample per class)
- 5-shot: Realistic clinical annotation budget
- 10-shot: More comfortable data regime

### 5. Few-Shot Learning Curves (Cells 14-15)

**Cell 14**: Markdown header
**Cell 15**: Line plot showing performance scaling

**Visualization:**
- One line per model showing AUC improvement from 1-shot → 5-shot → 10-shot
- Error bars showing confidence intervals
- Helps identify models with better low-data generalization

**Key Insight:**
- Steep curves = better generalization with limited data
- Shallow curves = data efficiency plateau

### 6. Alignment-Based Ensemble Section (Cells 16-17)

**Cell 16**: Markdown header and explanation
**Cell 17**: Code implementation

**What it does:**
1. Extracts all model features (train/val/test combined)
2. Evaluates ensemble across 10 splits
3. Weights models by mutual k-NN overlap
4. Compares ensemble performance to best individual model

**Key Output:**
```python
ensemble_model = {
    'individual_models': {model_name: knn_classifier},
    'weights': {model_name: weight},
    'model_list': [model_names]
}
```

**Performance Comparison:**
- Shows ensemble AUC vs each individual model
- Computes improvement over best single model
- Identifies which models contribute most

### 7. Ensemble Weights Visualization (Cells 18-19)

**Cell 18**: Markdown header
**Cell 19**: Bar chart and weight statistics

**Visualization:**
- Bar chart showing weight for each model
- Text labels on bars showing exact weights
- All weights sum to 1.0 (normalized)

**Weight Interpretation:**
- High weight (>0.15): Model aligns well with others (consensus)
- Medium weight (0.08-0.15): Balanced contribution
- Low weight (<0.08): Unique/diverse representation

---

## Results Storage

All results are stored in Python dictionaries for easy comparison:

```python
# KNN Results (pre-existing)
test_accuracies_dict[model_name] = {"mean": auc, "ci95": (lower, upper)}

# Linear Probing Results (new)
linear_probing_results[model_name] = {"mean": auc, "ci95": (lower, upper)}

# Few-Shot Results (new)
few_shot_results[shots][model_name] = {"mean": auc, "ci95": (lower, upper)}

# Ensemble Results (new)
ensemble_mean, ensemble_ci  # Overall ensemble performance
individual_ensemble_scores[model_name]  # Individual model performance within ensemble
```

---

## Expected Results for LUNA Dataset

Based on typical performance patterns:

### KNN vs Linear Probing
- Should have high correlation (r > 0.8)
- Linear Probing typically 0-3% higher than KNN
- High correlation validates feature quality

### Few-Shot Learning
Expected performance ranking: 10-shot > 5-shot > 1-shot
- Models show smooth degradation as shots decrease
- Well-designed models maintain >80% of full performance at 1-shot
- Steep curves indicate better low-data generalization

### Ensemble Method
- Ensemble typically performs ≥ best individual model
- Improvement depends on model diversity
- LUNA dataset models are likely complementary

---

## Notebook Structure

```
Cell 0-7:   Original KNN evaluation (unchanged)
Cell 8-9:   Linear Probing
Cell 10-11: Comparison (KNN vs Linear vs Few-Shot)
Cell 12-13: Few-Shot Learning evaluation
Cell 14-15: Few-Shot Learning curves
Cell 16-17: Ensemble Method
Cell 18-19: Ensemble weights visualization
Cell 20-22: Original overlap analysis (moved/unchanged)
```

---

## How to Run

1. **Linear Probing**: Run cells 8-9 after KNN evaluation completes
2. **Comparison**: Run cells 10-11 (requires both KNN and Linear results)
3. **Few-Shot**: Run cells 12-15 (can run in parallel with linear probing)
4. **Ensemble**: Run cells 16-19 (requires overlap_matrix from KNN section)

**Typical runtime:**
- Linear Probing: ~10-15 minutes
- Few-Shot Learning: ~20-30 minutes (3 shot configs × 10 splits)
- Ensemble: ~15-20 minutes

---

## Integration with Other Notebooks

The same methods can be added to other modelling notebooks:
- `nsclc_radiomics_modelling.ipynb`
- `nsclc_radiogenomics_modelling.ipynb`
- `colorectal_liver_metastases_modelling.ipynb`
- `c4c_kits_modelling.ipynb`
- `dlcs_modelling.ipynb`

Simply adapt the label key (e.g., `"malignancy"` → `"survival"` or `"label"`) based on your dataset.

---

## Addressing Reviewer Feedback

These additions directly address concerns from the Nature Communications review:

### Reviewer #2 - Expand Beyond KNN Probing
✅ Linear Probing - Standard transfer learning baseline
✅ Few-Shot Learning - Evaluation at 1, 5, 10-shot regimes
✅ Comparison Tables - All results easily compared

### Reviewer #1/#2 - Novel Methodology
✅ Alignment-Based Ensemble - Novel weighting scheme based on k-NN overlap

### Reviewer #3 - Demonstrate Innovation
✅ Multiple complementary evaluation protocols
✅ Data-driven ensemble approach
✅ Interpretable results through weight visualization

---

## Next Steps

1. Run the notebook to generate results
2. Collect results from all 6 datasets
3. Create comparison tables across datasets
4. Add results to manuscript
5. Update supplementary methods section

---

## Files Modified

- `luna_modelling.ipynb` - Updated with 12 new cells
- `modelling_utils.py` - Already contains all required functions

## Files Created/Referenced

- `ADVANCED_METHODS.md` - Full technical documentation
- `QUICK_REFERENCE.md` - Quick lookup guide
- `REVIEWER_FEEDBACK_RESPONSE.md` - Mapping to review comments

---

**Last Updated**: November 7, 2025
**Total New Cells Added**: 12
**Estimated Runtime**: ~45-65 minutes for all new evaluations
