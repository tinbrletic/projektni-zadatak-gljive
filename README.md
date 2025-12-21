# Feature Selection & Classification Pipeline: Mushroom Dataset

A comprehensive machine learning framework for **statistical feature selection** and **classifier evaluation** on binary classification tasks. This project implements multiple feature selection strategies (statistical tests, integrated selectors, RFECV) combined with 6 diverse classifiers, evaluated through repeated stratified k-fold cross-validation.

## 🎯 Project Overview

This pipeline addresses a core challenge in machine learning: **identifying the most informative features** for classification while avoiding data leakage. The script implements a rigorous, production-grade approach to feature selection with the following key characteristics:

- **Leak-proof design**: Feature selection is performed _inside_ the cross-validation loop, ensuring selection decisions are made only on training data
- **Statistical rigor**: Implements multiple statistical tests (Mann-Whitney U, Kruskal-Wallis, Chi-square, Kolmogorov-Smirnov, Wilcoxon) with multiple-testing correction (Benjamini-Hochberg, Holm)
- **Comprehensive baselines**: Baseline models trained on all features, statistical subsets, and integrated selectors
- **Repeated CV**: 10-fold cross-validation repeated 5 times (50 total iterations) for robust performance estimation
- **Out-of-fold (OOF) ROC**: Accumulates predictions across repeated CV folds to compute calibrated OOF probabilities and ROC curves
- **Extensible framework**: Easy to add new classifiers, feature selection methods, or datasets

## 📊 Dataset: UCI Mushroom Classification

**Source**: [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)

### Dataset Characteristics

- **Total Samples**: 8,124 mushroom observations
- **Target Classes**: 2 (edible "e" vs. poisonous "p") — **highly imbalanced** (~68% poisonous)
- **Features**: 22 categorical/numerical attributes including:
  - **Morphological**: cap shape, cap color, gill attachment, stalk shape, stalk color
  - **Biochemical**: odor, spore print color, veil type
  - **Physicochemical**: gill spacing, gill size, ring type, habitat
  - **Protein properties** (one-hot encoded): hydrophobic indices (Janin, Eisenberg, Kyte-Doolittle), charge, polarity groups, amino acid composition vectors

### Key Characteristics

- **Imbalanced**: Majority class (poisonous) ≈ 68%, minority (edible) ≈ 32%
- **Categorical**: Features are predominantly categorical; one-hot encoding is applied
- **Clean**: No missing values; well-structured dataset ideal for classification and feature selection studies

## 🛠️ Features

### Feature Selection Methods

1. **Statistical Tests** (in-CV)
   - Mann-Whitney U test (independent samples)
   - Kruskal-Wallis test (non-parametric alternative)
   - Chi-square test (categorical features)
   - Kolmogorov-Smirnov test (continuous distributions)
   - Wilcoxon test (paired/ranked)
2. **Integrated Selectors** (fitted as pipeline steps)

   - `SelectFromModel` with L1-penalized Logistic Regression
   - `SelectFromModel` with Random Forest
   - `RFECV` (Recursive Feature Elimination with Cross-Validation)

3. **Baseline**
   - All features (no selection)

### Classifiers

1. **Random Forest** (200 estimators, max_depth=10)
2. **Logistic Regression** (L1 penalty, balanced classes)
3. **Support Vector Machine** (RBF kernel, probability calibrated)
4. **K-Nearest Neighbors** (k=15, KD-tree)
5. **Decision Tree** (max_depth=5)
6. **Gaussian Naive Bayes** (with standard scaling)

Each classifier is evaluated with:

- All features (baseline)
- Top-5 features from each statistical test (5 variants = 5 models per classifier)
- Integrated selectors (3 variants = 3 more models per classifier)

**Total**: 6 base classifiers × (1 + 5 + 3) = 54 configurations

### Evaluation Metrics

- **Per-fold**: Accuracy, F1, Precision, Recall, MCC, AUC-ROC
- **Aggregated**: Mean ± Std across 50 CV iterations
- **Statistical Tests**: Friedman test, pairwise Wilcoxon, Nemenyi post-hoc tests
- **OOF Diagnostics**: Confusion matrices, ROC curves, AUC values

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.0.0
matplotlib>=3.4.0
scikit-posthocs>=0.5.0 (optional, for advanced post-hoc tests)
skrebate>=0.6 (optional, for ReliefF selector)
```

Install with:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Dataset

Place `mushrooms.csv` in the project root directory.

```bash
wget https://www.kaggle.com/datasets/uciml/mushroom-classification/download
unzip mushroom-classification.zip
```

### 2. Run the Pipeline

```bash
python feature_selection.py
```

The script will:

1. Load and preprocess the mushroom dataset
2. Perform statistical feature selection (cached for speed)
3. Train 54 classifier configurations across 50 CV folds
4. Cache results to avoid re-running
5. Generate performance reports, ROC plots, and CSV summaries

### 3. Access Results

Output files (generated with timestamp):

- `feature_selection_results_mushrooms.csv_YYYYMMDD_HHMMSS.txt` — Detailed results report
- `feature_selection_metrics_YYYYMMDD_HHMMSS.csv` — Performance metrics table
- `selected_features_summary_YYYYMMDD_HHMMSS.csv` — Selected features by method
- `stats_*.csv` / `stats_*.json` — Statistical comparison results
- `roc_*.png` — ROC curve plots (aggregated and per-model)

### 4. Skip Cache (Re-run)

To skip cached results and re-run from scratch:

```bash
NO_CACHE=1 python feature_selection.py
```

Or:

```bash
python feature_selection.py --no-cache
```

## 📁 Project Structure

```
├── feature_selection.py          # Main pipeline script
├── mushrooms.csv                 # Dataset (download required)
├── cache/                        # Auto-generated: cached results & splits
│   ├── joblib/                   # Joblib memory cache for statistical tests
│   ├── dataset_fingerprint.json  # Dataset hash for change detection
│   ├── results_*.pkl             # Pickled classifier results
│   ├── featimp_*.pkl             # Pickled feature importances
│   └── oof_*.npz                 # Out-of-fold predictions
├── feature_selection_results_*.txt
├── feature_selection_metrics_*.csv
├── selected_features_summary_*.csv
└── roc_*.png
```

## 🔬 Key Design Decisions

### 1. Repeated Cross-Validation

- **10-fold × 5 repeats** (50 iterations) to account for variance across random splits
- Ensures robust and stable performance estimates

### 2. In-CV Feature Selection

- Feature selection is performed **inside** each fold's training data only
- Avoids data leakage and provides realistic feature importance estimates
- `StatSelect` transformer integrates seamlessly into scikit-learn pipelines

### 3. OOF Predictions for Repeated CV

- Accumulates test predictions across all 50 folds using `y_oof_sum` and `y_oof_cnt`
- Computes averaged OOF probabilities: `y_oof[i] = y_oof_sum[i] / y_oof_cnt[i]`
- Enables proper ROC/AUC computation for repeated CV without leakage

### 4. Multiple Testing Correction

- Controls false positives in statistical tests
- Benjamini-Hochberg (FDR) and Holm correction options
- Effect size filtering (minimum effect size threshold: 0.10)

### 5. Caching

- Joblib memory cache for expensive statistical feature selection
- Fingerprint-based dataset detection for Parquet conversion
- Split cache ensures deterministic CV splits across runs

## 📈 Example Results

### Performance Summary (Baseline: All Features)

| Classifier          | Accuracy      | F1-Score      | Precision     | Recall        | AUC-ROC       |
| ------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Random Forest       | 0.868 ± 0.013 | 0.928 ± 0.008 | 0.882 ± 0.014 | 0.979 ± 0.013 | 0.721 ± 0.025 |
| K-Neighbors         | 0.873 ± 0.014 | 0.931 ± 0.008 | 0.882 ± 0.014 | 0.986 ± 0.010 | 0.692 ± 0.048 |
| Naive Bayes         | 0.819 ± 0.020 | 0.894 ± 0.013 | 0.911 ± 0.011 | 0.878 ± 0.026 | 0.734 ± 0.033 |
| Logistic Regression | 0.707 ± 0.043 | 0.809 ± 0.035 | 0.925 ± 0.008 | 0.721 ± 0.052 | 0.705 ± 0.046 |

### Most Discriminative Features

1. **Hydrophobic Indices** (Janin, Eisenberg, Kyte-Doolittle) — strong signal
2. **Non-polar & Polar Groups** — compositional differences
3. **Charge & Acidity** — biochemical distinctions
4. **Peptide Length** — physical characteristic

## 🔍 Usage Tips

### Add a New Classifier

Modify the `classifiers` dictionary in the script:

```python
classifiers["New Model (All)"] = {
    'model': YourNewClassifier(params...),
    'selector': None,
    'features': all_features
}
```

### Adjust Feature Selection Parameters

Modify `TOP_K` and `StatSelect` hyperparameters:

```python
TOP_K = 10  # Select top 10 features instead of 5
# In classifiers dict:
StatSelect(method='mann_whitney', top_k=TOP_K, alpha=0.05, ...)
```

### Change Cross-Validation Strategy

Modify the `kf` object:

```python
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
```

## 📚 References & Methods

- **Statistical Tests**: SciPy documentation (mannwhitneyu, kruskal, chi2_contingency, ks_2samp)
- **Multiple Testing Correction**: Benjamini & Hochberg (1995), Holm (1979)
- **Feature Selection**: Scikit-learn SelectFromModel, RFECV, sklearn.feature_selection
- **Cross-Validation**: Repeated Stratified K-Fold as per scikit-learn recommendations
- **Statistical Comparisons**: Friedman test with Nemenyi post-hoc (Demšar, 2006)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Suggestions for improvements:

- Add new feature selection methods (e.g., Mutual Information, Permutation Importance)
- Implement hyperparameter tuning (GridSearchCV, Optuna)
- Extend to multi-class classification
- Add ensemble methods (Voting, Stacking)
- Implement SHAP or LIME for explainability

---

**Questions?** Refer to the inline documentation in `feature_selection.py` and the generated output files.
