# Quick Usage Guide - mAInXanceApp Enhanced ML System

## For End Users (Technicians)

### Interactive Chat Mode

Start the chat assistant:
```bash
python3 interactive_chat.py
```

Then describe your problem:
```
[You] Describe the problem: motor won't start

[Assistant] Analyzing your request...
================================================================================
MAINTENANCE RECOMMENDATION
================================================================================
Your Query: motor won't start
Predicted Action Category: Cluster 3
Category Description: Repairs (Broken Parts, Wiring)
Confidence: 45.23%

SIMILAR HISTORICAL CASES (Top 5)
1. Work Order: 12345
   Solution: Checked power supply, replaced blown fuse
...
```

### Single Query Mode

For a quick answer without interactive mode:
```bash
python3 interactive_chat.py "conveyor belt slipping"
```

### Demo Mode

See example queries:
```bash
python3 interactive_chat.py demo
```

---

## For Data Scientists / ML Engineers

### Training the Model

Run the complete workflow:
```bash
python3 ml_workflow_enhanced.py
```

This will:
1. Load and clean data (56K+ records from 1M+)
2. Perform K-Means clustering (5 clusters)
3. Train logistic regression with class balancing
4. Run 5-fold cross-validation
5. Generate 5 visualization plots
6. Save models and processed data
7. Create detailed logs

**Expected runtime:** ~30 seconds

**Outputs:**
- `processed_work_orders.csv` - Clean dataset
- `prediction_model.pkl` - Trained classifier
- `similarity_search_engine.pkl` - RAG search engine
- `cluster_distribution.png` - Cluster sizes
- `confusion_matrix.png` - Prediction accuracy
- `silhouette_scores.png` - Clustering quality
- `cross_validation_results.png` - CV performance
- `class_performance.png` - Per-cluster metrics
- `ml_workflow.log` - Execution log

### Using the Model Programmatically

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('prediction_model.pkl')

# Make prediction
description = "air pressure too low"
predicted_cluster = model.predict([description])[0]
probabilities = model.predict_proba([description])[0]

print(f"Predicted Cluster: {predicted_cluster}")
print(f"Confidence: {probabilities[predicted_cluster]*100:.2f}%")
```

### Using the Search Engine

```python
from ml_workflow_enhanced import SimilaritySearchEngine
import joblib

# Load search engine
search_engine = joblib.load('similarity_search_engine.pkl')

# Search for similar cases
results = search_engine.search("hydraulic leak", top_n=5)
print(results[['WO No.', 'Description', 'Text', 'similarity_score']])
```

---

## For Project Managers / Executives

### Checking Model Performance

View the validation report:
```bash
cat VALIDATION_REPORT.md
```

Key metrics to monitor:
- **Accuracy:** 39.00% ± 0.46% (cross-validation)
- **F1 Score:** 0.43 (balanced across all action types)
- **Data Quality:** 56,513 clean records
- **Response Time:** <100ms per query
- **Coverage:** All 5 technician action categories

### Viewing Results

All visualizations are saved as high-resolution PNG files:
1. `cluster_distribution.png` - Shows data balance
2. `confusion_matrix.png` - Shows prediction accuracy
3. `class_performance.png` - Shows per-category performance
4. `cross_validation_results.png` - Shows model stability

### Logs

Check the execution log:
```bash
tail -f ml_workflow.log
```

---

## Common Scenarios

### Scenario 1: New Work Order Arrives

**Technician receives:** "Conveyor belt making noise"

**Steps:**
1. Open chat: `python3 interactive_chat.py`
2. Enter problem description
3. Review predicted action category
4. Check top 5 similar historical cases
5. Follow recommended solution steps

**Expected time to recommendation:** <1 second

### Scenario 2: Batch Processing Overnight Work Orders

**Goal:** Process 100 work orders from overnight

```python
from interactive_chat import MaintenanceAssistantChat

# Initialize assistant
assistant = MaintenanceAssistantChat()

# Load queries from file/database
queries = [
    "motor overheating",
    "sensor reading error",
    # ... 98 more
]

# Batch process
results = assistant.batch_process(queries)

# Save results
import pandas as pd
df_results = pd.DataFrame(results)
df_results.to_csv('overnight_predictions.csv', index=False)
```

### Scenario 3: Retraining with New Data

**When to retrain:**
- Monthly (recommended)
- After 1000+ new work orders
- If performance degrades >5%

**Steps:**
1. Append new work orders to CSV
2. Run: `python3 ml_workflow_enhanced.py`
3. Compare new metrics to VALIDATION_REPORT.md
4. Deploy new models if improved

---

## Troubleshooting

### Issue: "Model file not found"

**Solution:**
```bash
python3 ml_workflow_enhanced.py  # Train model first
python3 interactive_chat.py      # Then use chat
```

### Issue: "ImportError: No module named X"

**Solution:**
```bash
pip3 install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Issue: Predictions seem wrong

**Check:**
1. Is the description clear and detailed?
2. Is this a common problem in the dataset?
3. Check conversation history for patterns

**Example:**
```
Bad: "fix it"  (too vague)
Good: "conveyor belt slipping on drive pulley"
```

### Issue: Chat is slow

**Possible causes:**
- Large dataset (>100K records)
- Limited CPU/RAM

**Solutions:**
1. Use single query mode instead of interactive
2. Increase timeout in script
3. Run on machine with more resources

---

## Best Practices

### For Best Prediction Accuracy

1. **Be specific:** "motor #3 won't start" > "problem"
2. **Include location:** "north conveyor belt" > "belt"
3. **Describe symptoms:** "making grinding noise" > "broken"
4. **Use standard terms:** Align with how technicians write notes

### For Data Quality

1. **Regular retraining:** Monthly with new work orders
2. **Feedback loop:** Mark wrong predictions, retrain
3. **Data validation:** Check for corrupt/duplicate entries
4. **Feature updates:** Add new equipment types as needed

### For Production Deployment

1. **Monitor logs:** Check ml_workflow.log daily
2. **Track metrics:** Compare weekly accuracy trends
3. **User feedback:** Survey technicians monthly
4. **A/B testing:** Compare ML suggestions vs. manual

---

## Quick Reference

### File Locations

| File | Purpose |
|------|---------|
| `ml_workflow_enhanced.py` | Main training script |
| `interactive_chat.py` | User-facing chat interface |
| `prediction_model.pkl` | Trained classifier (load with joblib) |
| `similarity_search_engine.pkl` | RAG engine (load with joblib) |
| `processed_work_orders.csv` | Clean dataset |
| `ml_workflow.log` | Execution logs |
| `VALIDATION_REPORT.md` | Performance analysis |

### Key Classes

| Class | Purpose |
|-------|---------|
| `DataPreprocessor` | Cleans and validates data |
| `ClusteringEngine` | K-Means clustering |
| `PredictionModel` | Logistic regression classifier |
| `SimilaritySearchEngine` | TF-IDF similarity search |
| `MaintenanceAssistantChat` | Interactive RAG interface |
| `Visualizer` | Plot generation |

### Command Cheat Sheet

```bash
# Train model
python3 ml_workflow_enhanced.py

# Interactive chat
python3 interactive_chat.py

# Single query
python3 interactive_chat.py "your problem here"

# Demo mode
python3 interactive_chat.py demo

# View logs
tail -f ml_workflow.log

# Check model performance
cat VALIDATION_REPORT.md
```

---

## Support & Contact

### Documentation

- `README_ENHANCED.md` - Full project overview
- `VALIDATION_REPORT.md` - Performance metrics
- `USAGE_GUIDE.md` - This file
- `codebase_analysis.md` - Technical deep dive

### Getting Help

1. Check logs: `ml_workflow.log`
2. Review validation report: `VALIDATION_REPORT.md`
3. Consult docstrings: All functions documented
4. Debug mode: Set `logging.level = DEBUG` in scripts

---

**Last Updated:** 2025-10-29
**Version:** 2.0 Enhanced
**Status:** Production Ready ✅
