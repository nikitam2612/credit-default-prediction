# Credit Default Prediction

## Project Overview
This project predicts whether a credit card client is likely to default on their next payment using demographic information, credit limits, repayment history, bill amounts, and previous payment amounts.

The project demonstrates a basic machine learning workflow, including data quality checks, exploratory data analysis (EDA), model comparison, threshold tuning, and feature importance analysis.

## Business Objective
Credit default prediction can help financial institutions identify customers who may require additional credit-risk assessment. The objective is to explore how machine learning can distinguish between customers who default and those who do not.

## Dataset
- **Source:** UCI Machine Learning Repository — Default of Credit Card Clients
- **Records:** 30,000
- **Columns:** 25, including the ID and target columns
- **Target:** `default payment next month`
- **Target values:** `0` = no default; `1` = default

The dataset contains an imbalanced target distribution:
- No default: 23,364 (77.88%)
- Default: 6,636 (22.12%)

## Tools and Technologies
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Excel dataset (`.xls`)

## Project Workflow
1. Load the dataset and validate its structure.
2. Check missing values, duplicates, target values, and categorical variables.
3. Explore target distribution and relationships between customer attributes and default.
4. Train a class-weighted Logistic Regression baseline.
5. Compare Logistic Regression with Random Forest.
6. Tune the classification threshold using validation data.
7. Examine Random Forest feature importance.

## Models and Results

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6797 | 0.3672 | 0.6202 | 0.4613 | 0.7081 |
| Random Forest | 0.8005 | 0.5562 | 0.4846 | 0.5179 | 0.7606 |

In this initial comparison, Random Forest achieved higher accuracy, precision, F1-score, and ROC-AUC, while Logistic Regression achieved higher recall.

### Threshold Tuning

A Random Forest model was evaluated using separate training, validation, and test partitions. A threshold of **0.45** was selected from the tested thresholds based on validation F1-score.

Test-set results at the selected threshold:

| Metric | Score |
|---|---:|
| Accuracy | 0.7853 |
| Precision | 0.5142 |
| Recall | 0.5328 |
| F1-score | 0.5233 |
| ROC-AUC | 0.7635 |

These threshold-tuning results come from a different train/test partition than the initial model comparison. They should therefore be interpreted as a separate evaluation, not a direct before-and-after comparison.

## Key Findings
- The dataset has a class imbalance, with default cases representing 22.12% of observations.
- Repayment-status variables, particularly `PAY_0`, are useful signals in the trained Random Forest model.
- Random Forest performed better than Logistic Regression on most metrics in the initial comparison, although Logistic Regression had higher recall.
- Classification threshold selection affects the balance between precision and recall.

## Project Outputs
- `outputs/figures/` — exploratory data analysis charts and feature importance visualization.
- `outputs/metrics/` — model metrics, model comparison results, threshold-tuning results, and feature importance data.

## Limitations
- The results are based on one dataset and particular train/validation/test splits.
- Feature importance indicates model associations, not causation.
- Classification metrics do not establish that a model is suitable for real-world credit decisions.
- Further validation, calibration, fairness assessment, and cost-sensitive evaluation would be required before any practical deployment.

## How to Run
Create and activate a Python virtual environment, install the dependencies, and run the scripts from the project root.

```bash
pip install -r requirements.txt
python src/data_loader.py
python src/data_quality.py
python src/eda.py
python src/train_model.py
python src/compare_models.py
python src/tune_threshold.py
python src/feature_importance.py
```

## Disclaimer
This project is for educational and portfolio purposes only. It is not intended for making real-world lending or credit approval decisions.
