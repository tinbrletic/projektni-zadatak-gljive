# Feature Selection & Classification on Mushroom Dataset

A machine learning pipeline that evaluates **multiple feature selection methods** combined with **6 different classifiers** on binary classification, using rigorous repeated cross-validation.

## 📌 Purpose & Goal

This project demonstrates a **production-grade approach to feature selection** that:

- **Avoids data leakage** by performing feature selection inside the CV loop (train data only)
- **Compares multiple strategies**: statistical tests, model-based selectors, and full-feature baselines
- **Provides robust evaluation**: 10-fold CV repeated 5 times (50 total iterations) for stable metrics
- **Implements statistical rigor**: Multiple testing correction (Benjamini-Hochberg, Holm), effect size filtering

## 🎯 What Does This Script Do?

1. **Loads mushroom classification dataset** from CSV
2. **Performs statistical feature selection** using 5 methods:
   - Mann-Whitney U test
   - Kruskal-Wallis test
   - Chi-square test
   - Kolmogorov-Smirnov test
   - Wilcoxon test (with Benjamini-Hochberg FDR correction)
3. **Trains 6 diverse classifiers** with:
   - All features (baseline)
   - Top-5 features from each statistical test (in-CV to avoid leakage)
   - Integrated selectors (SelectFromModel, RFECV)
4. **Evaluates performance** across 50 CV folds using:
   - Accuracy, F1, Precision, Recall, AUC-ROC, MCC
   - ROC curves from out-of-fold (OOF) predictions
   - Statistical tests (Friedman, Wilcoxon pairwise)
5. **Generates detailed reports** with results, metrics, and visualizations

## 📊 Dataset: UCI Mushroom Classification

**Source**: [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)

### Key Characteristics

- **Samples**: 8,124 mushroom observations
- **Classes**: Binary (edible "e" vs. poisonous "p") — ~68% poisonous, ~32% edible
- **Features**: 22 categorical attributes (cap shape, color, odor, gill properties, stalk properties, etc.)
- **Quality**: No missing values; clean, well-structured dataset

## ⚡ Quick Start

### Requirements

```bash
pip install pandas numpy scikit-learn scipy joblib matplotlib
```

### Run

```bash
# Download dataset from Kaggle and place in project root as 'mushrooms.csv'
python feature_selection.py
```

### Output

- `feature_selection_results_*.txt` — Detailed performance report
- `feature_selection_metrics_*.csv` — Performance metrics table
- `stats_*.csv` — Statistical comparison results
- `roc_*.png` — ROC curve plots

### Skip Cache (Re-run)

```bash
python feature_selection.py --no-cache
```

## 🛠️ Key Features

### Feature Selection Methods

1. **Statistical Tests** (5 methods, top-5 features each)

   - Applied inside CV loop → no data leakage
   - Multiple testing correction
   - Effect size filtering (ES ≥ 0.10)

2. **Integrated Selectors** (fitted as pipeline steps)

   - SelectFromModel with L1-LR
   - SelectFromModel with Random Forest
   - RFECV (Recursive Feature Elimination)

3. **Baseline**
   - All features (no selection)

### Classifiers (6 total)

- Random Forest
- Logistic Regression (L1 penalty)
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes

Each tested with all feature selection strategies.

## 📈 Evaluation Strategy

- **Cross-Validation**: RepeatedStratifiedKFold (10 splits × 5 repeats = 50 iterations)
- **Metrics**: Accuracy, F1, Precision, Recall, MCC, AUC-ROC, Confusion Matrix
- **OOF Diagnostics**: Out-of-fold predictions averaged across repeated CV folds
- **Statistical Comparison**: Friedman test, pairwise Wilcoxon with Holm correction

## 🔍 Design Highlights

✅ **Leak-proof**: Feature selection performed only on training data within CV folds  
✅ **Statistical integrity**: Multiple testing correction prevents false positives  
✅ **Reproducible**: Cached results with dataset fingerprinting; deterministic splits  
✅ **Robust evaluation**: Repeated CV for stable performance estimates  
✅ **Extensible**: Easy to add new classifiers or feature selection methods
