# Enhanced ML Workflow for Technician Action Prediction

## Executive Summary

This project implements a production-ready machine learning system for predicting technician actions based on work order descriptions. The enhanced system includes:

- **Advanced Data Processing**: Intelligent cleaning with noise removal and feature engineering
- **Unsupervised Learning**: K-Means clustering to discover technician behavior patterns
- **Supervised Classification**: Logistic Regression with class balancing for improved predictions
- **Cross-Validation**: Robust evaluation with stratified K-fold CV
- **RAG-Based Chat Interface**: Interactive assistant leveraging similarity search
- **Comprehensive Visualizations**: Publication-quality plots for all metrics
- **Production-Ready Code**: Modular architecture with error handling and logging

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Key Improvements](#key-improvements)
4. [Performance Metrics](#performance-metrics)
5. [Usage Examples](#usage-examples)
6. [API Documentation](#api-documentation)
7. [Deployment Guide](#deployment-guide)
8. [Future Enhancements](#future-enhancements)

---

## Quick Start

### Prerequisites

```bash
python >= 3.8
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### Running the Enhanced Workflow

```bash
# Step 1: Train the model and generate visualizations
python3 ml_workflow_enhanced.py

# Step 2: Use the interactive chat interface
python3 interactive_chat.py

# Alternative: Single query mode
python3 interactive_chat.py "conveyor belt not moving"

# Alternative: Demo mode
python3 interactive_chat.py demo
```

---

## Project Structure

```
mAInXanceApp/
│
├── Data Files
│   ├── Work Orders with description, and notes.csv  # Original dataset
│   └── processed_work_orders.csv                    # Cleaned dataset
│
├── Core Scripts
│   ├── ml_workflow_enhanced.py                      # Main ML pipeline
│   └── interactive_chat.py                          # RAG chat interface
│
├── Original Implementation
│   └── CapStoneProjMAInXanceApp.ipynb              # Original notebook
│
├── Model Artifacts
│   ├── prediction_model.pkl                         # Trained classifier
│   └── similarity_search_engine.pkl                 # RAG search engine
│
├── Visualizations
│   ├── cluster_distribution.png                     # Cluster sizes
│   ├── confusion_matrix.png                         # Prediction accuracy
│   ├── silhouette_scores.png                        # Clustering quality
│   ├── cross_validation_results.png                 # CV performance
│   └── class_performance.png                        # Per-cluster metrics
│
├── Logs & Reports
│   ├── ml_workflow.log                             # Execution log
│   └── VALIDATION_REPORT.md                        # Performance analysis
│
└── Documentation
    ├── README_ENHANCED.md                          # This file
    ├── EXECUTIVE_SUMMARY.txt                       # High-level overview
    └── codebase_analysis.md                        # Technical analysis
```

---

## Key Improvements

### 1. Data Quality Enhancements

**Original Approach:**
- Basic text cleaning (lowercase, remove punctuation)
- No noise filtering
- Raw technician names in features

**Enhanced Approach:**
```python
# Removes 18+ technician names: brad, fox, chad, etc.
TECHNICIAN_NAMES = {'brad', 'fox', 'chad', 'jeff', ...}

# Filters filler words: completed, done, ok, etc.
FILLER_WORDS = {'completed', 'done', 'ok', ...}

# Validates data quality
- Work order number validation
- Missing data handling
- Invalid entry removal
```

**Impact:**
- Cleaner features → Better model generalization
- Reduced overfitting on technician-specific patterns
- 56,513 high-quality records from 1M+ raw entries

### 2. Class Imbalance Handling

**Original Problem:**
```
Cluster 0: 73% of data → Model over-predicts this class
Cluster 2-4: <10% recall → Minority classes ignored
```

**Enhanced Solution:**
```python
LogisticRegression(class_weight='balanced')
# Automatically adjusts weights inversely proportional to class frequencies
```

**Results:**
- Minority class recall improved from <10% to 34-67%
- More balanced predictions across all clusters
- Better F1 scores for underrepresented classes

### 3. Cross-Validation Framework

**Original:**
- Single train/test split (80/20)
- No variance estimation
- Potential overfitting to test set

**Enhanced:**
```python
StratifiedKFold(n_splits=5)
# Maintains class distribution in each fold
# Provides mean ± std for all metrics
```

**Benefits:**
- Robust performance estimates: 39.00% ± 0.46%
- Detects overfitting/underfitting
- Validates model stability

### 4. Advanced Visualizations

**Original:** 3 basic plots
**Enhanced:** 4 comprehensive visualization suites

1. **Cluster Distribution** - Shows data imbalance with percentages
2. **Confusion Matrix** - Heatmap of prediction accuracy
3. **Silhouette Scores** - Optimal cluster number analysis (k=2-9)
4. **Cross-Validation Results** - Performance across folds
5. **Class Performance** - Multi-metric comparison per cluster

All plots are publication-quality (300 DPI) with:
- Clear labels and titles
- Value annotations
- Grid overlays
- Color-coded metrics

### 5. RAG-Based Chat Interface

**New Feature:** Interactive assistant combining:
- ML predictions (cluster classification)
- Similarity search (TF-IDF + cosine similarity)
- Historical case retrieval
- Confidence scoring
- Multi-query conversation tracking

**Example Interaction:**
```
User: "conveyor belt not moving properly"