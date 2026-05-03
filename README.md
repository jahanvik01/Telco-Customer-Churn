# Telco Customer Churn — Analysis & Prediction

> **End-to-end churn analysis on the IBM Telco dataset (7,043 customers), going from raw data to ROI-tagged retention plays worth ~$813k/year.**

A portfolio data analyst project demonstrating EDA, cohort segmentation, predictive modeling (Logistic Regression + Random Forest), threshold tuning to a business cost function, SHAP interpretation, and concrete business recommendations.

---

## Problem Statement

A telecom company loses **~26.5% of its customers each year**. Acquiring a replacement costs 5–25× more than retaining one (HBR/Bain). The business wants to know:

1. Which customer segments churn most?
2. What are the strongest churn predictors?
3. Can we predict churn well enough to act on it?
4. What concrete retention plays should the business run, and what's the ROI?

This project answers all four.

---

## Headline Findings

- **The whole churn problem lives in one cell:** *Month-to-month contract + Fiber optic + 0–12 months tenure* — 916 customers, **70% churn rate**, ~**$633k/year revenue at risk**.
- **Quadruple agreement on top drivers** (EDA, LR coefficients, RF importance, SHAP):
  1. Contract type — M2M ~43% vs Two-year ~3% (14× gap)
  2. Tenure — first 12 months are the danger window
  3. Internet service — Fiber customers churn at ~42% (premium product, worst retention)
  4. Payment method — Electronic-check users churn at ~45%
  5. Protection bundle — having 0 protection services doubles churn vs having 4
- Demographics (`gender`) carry **no signal** and were dropped.

---

## Methodology

| Stage | What I did |
|---|---|
| **1. Cleaning** | Fixed `TotalCharges` (string with 11 blanks for tenure-0 customers); collapsed redundant "No internet/phone service" levels; encoded binary Yes/No → 0/1. |
| **2. EDA** | One chart + one written insight per major feature. Identified the top 5 churn drivers visually. |
| **3. Cohort analysis** | Heatmaps of Tenure × Contract and Internet × Contract. Engineered `protection_count` (0–4 protection services) — clean monotonic signal with churn. Translated cohorts to dollars at risk. |
| **4. Feature engineering** | 31 final features: original numerics, binary 0/1 columns, 4 risk flags (`is_fiber`, `is_month_to_month`, `is_electronic_check`, `has_no_protection`), one-hot encoded multi-class with `drop_first=True`. |
| **5. Modeling** | Stratified 80/20 split. Two models: Logistic Regression (scaled, interpretable) and Random Forest (non-linear, predictive). Both with `class_weight='balanced'`. |
| **6. Evaluation** | Confusion matrices, ROC + PR curves. Cost-tuned threshold using FN=$1000, FP=$50 cost asymmetry. |
| **7. Interpretation** | LR coefficients, RF feature importances, SHAP beeswarm + per-customer waterfalls. |
| **8. Recommendations** | 5 retention plays with target segment, cost-per-customer, expected churn reduction, and net ROI. |

---

## Results

### Model performance (held-out test set, n=1,409)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.742 | 0.509 | 0.786 | 0.618 | 0.841 |
| Random Forest (default 0.5 threshold) | 0.766 | 0.542 | 0.767 | 0.635 | 0.845 |
| **Random Forest (cost-tuned threshold = 0.08)** | 0.348 | 0.334 | **0.992** | 0.500 | 0.845 |

**Why the cost-tuned model wins:** with FN $1000 vs FP $50, missing churners is 20× more expensive than over-flagging. The cost-tuned threshold catches 99% of churners and minimizes total business cost ($40k vs $80k at default 0.5).

### 5 Retention Plays — expected annual ROI

| # | Play | Target | Cost | Net ROI |
|---|---|---:|---:|---:|
| 1 | Contract upgrade bonus (M2M+Fiber+0-12mo → 1yr) | 916 | $46k | **$226k** |
| 2 | Protection bundle trial (no-protection + tenure<24) | 1,752 | $53k | **$47k** |
| 3 | Auto-pay migration credit (electronic-check users) | 709 | $14k | **$150k** |
| 4 | First-year onboarding program | 2,186 | $66k | **$8k** |
| 5 | Fiber service investigation | 3,096 | $22k | **$383k** |
| | **Total** | | **$199k** | **$814k/year** |

---

## Repo Structure

```
TelcoChurnAnalysis/
├── data/
│   ├── raw/                    # original Kaggle CSV (gitignored)
│   └── processed/              # cleaned + feature-engineered datasets
├── notebooks/
│   └── churn_analysis.ipynb    # the main deliverable — 10 sections
├── models/
│   ├── logreg.pkl              # trained logistic regression
│   ├── rf.pkl                  # trained random forest
│   ├── scaler.pkl              # fitted StandardScaler
│   ├── splits.pkl              # X/y train/test splits (reproducibility)
│   └── decision_threshold.pkl  # cost-tuned threshold + cost assumptions
├── reports/
│   ├── figures/                # 22 saved PNG charts
│   └── retention_plays_roi.csv # final ROI table
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

```bash
# 1. Clone the repo and place the dataset
#    Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    Save as: data/raw/telco_churn.csv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/churn_analysis.ipynb

# 4. Kernel → Restart & Run All
```

The notebook is fully reproducible end-to-end. All intermediate files (cleaned data, feature matrix, trained models, charts) regenerate.

---

## Tech Stack

- **Python 3.13** — pandas 3.0, numpy, scikit-learn 1.7, matplotlib, seaborn, shap 0.51, joblib
- **Modeling:** Logistic Regression + Random Forest with `class_weight='balanced'`
- **Interpretation:** SHAP TreeExplainer
- **Threshold tuning:** custom cost function (FN=$1000, FP=$50)

---

## If I Had More Time

1. **Hyperparameter tuning** — Grid-search both models with cross-validation (likely +1-2 pp AUC).
2. **Try gradient boosting** — LightGBM or XGBoost usually edges out RF on tabular data.
3. **Calibrate probabilities** — `CalibratedClassifierCV` so `predict_proba` is honest.
4. **Time-aware split** — IBM data is a snapshot; production needs train-on-past, test-on-future split.
5. **Survival analysis** — predict not just *if* but *when* a customer churns (Cox PH).
6. **A/B test the recommendations** — run each play on a 10% holdout before scaling.
7. **Build a Streamlit dashboard** — customer ID lookup → per-customer SHAP waterfall → recommended play.
8. **Connect to live data** — hook into the company CRM via API.

---

## Author

**Jahanvi Kashyap** — data analyst portfolio project, May 2026.

Dataset: [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — copyright IBM, used here for educational purposes.
