# Credit Card Fraud Detection

Detecting fraudulent card transactions in a dataset where only **0.17% of transactions are fraud** (492 of 284,807). The goal was not high accuracy, which a model can get by predicting "not fraud" every time, but catching as much fraud as possible while keeping false alarms low enough that a fraud team could actually review them.

<img width="1389" height="590" alt="rf_roc_pr_curves" src="https://github.com/user-attachments/assets/2fdb7f16-b4ce-446e-9207-24b1a930017c" />

## Results (held-out test set, 56,962 transactions, 98 fraud)

| Model | ROC-AUC | PR-AUC | Fraud caught | False alarms | Precision |
|------------|------------|------------|------------|------------|------------|
| Logistic Regression + SMOTE | 0.971 | 0.72 | 90 / 98 | 1,465 | 0.06 |
| **Random Forest + SMOTE** | **0.984** | 0.80 | 86 / 98 | **57** | **0.60** |
| XGBoost + SMOTE | 0.980 | **0.83** | 88 / 98 | 241 | 0.27 |

*Default 0.5 threshold. Cross-validation (stratified k-fold on the training set) gave consistent numbers, e.g. PR-AUC 0.83 for Random Forest and 0.82 for XGBoost.*

**What this means in practice:** Logistic Regression catches the most fraud but would send an analyst about 16 false alarms for every real case. Random Forest catches nearly as many (86 vs 90) with roughly 25× fewer false alarms, which makes it the most workable model for a manual review queue.

## Why PR-AUC, not accuracy

With 99.83% legitimate transactions, a model that flags nothing scores 99.8% accuracy. PR-AUC measures how well the model ranks the rare fraud cases above normal ones, so it's the main metric here, with recall and precision on the fraud class alongside it.

## Approach

1.  **EDA:** class balance, transaction amount distributions for fraud vs. normal, and a correlation check across the PCA features (V1–V28).
2.  **Stratified train/test split (80/20)** so both sets keep the 0.17% fraud rate.
3.  **Imbalance handling with SMOTE inside an `imblearn` pipeline,** so synthetic samples are generated only within each training fold and never leak into validation or test data.
4.  **Models:** Logistic Regression (scaled), Random Forest, and XGBoost.
5.  **Evaluation:** stratified cross-validation on the training set, then a final check on the untouched test set using ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrices.
6.  **Threshold analysis:** plotted precision–recall trade-offs to show how moving the decision threshold changes the number of false alarms vs. missed fraud.

## Dataset

[Credit Card Fraud Detection (ULB Machine Learning Group, Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): transactions by European cardholders over two days. Features V1–V28 are PCA-transformed for confidentiality; `Time` and `Amount` are raw.

## How to run

``` bash
pip install -r requirements.txt
jupyter notebook credit_card_fraud_detection.ipynb
```

The notebook downloads the data automatically with `kagglehub` (a free Kaggle account/API token is needed the first time).

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · Matplotlib · Seaborn

## Next steps

-   Tune the decision threshold on a separate validation split and translate it into cost terms (cost of a missed fraud vs. cost of reviewing a false alarm).
-   Use a time-based split (train on earlier transactions, test on later ones) to better mimic real deployment.
-   Add SHAP explanations for individual flagged transactions.

------------------------------------------------------------------------

**Author:** Indraneel Mannava · [LinkedIn](https://www.linkedin.com/in/indraneel-sarma-mannava/)
