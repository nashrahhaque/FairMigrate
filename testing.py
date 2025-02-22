#!/usr/bin/env python3
# small_test_pipeline_sample_weight.py
# -------------------------------------------------------------------
# A minimal pipeline test script with a small sample (20 rows), 
# verifying that ExponentiatedGradient doesn't require `sensitive_features`
# during predict() in some Fairlearn versions.
# Outputs go to "my_output_small/" without printing to stdout.
# -------------------------------------------------------------------

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Fairlearn
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

# ------------------------- 1. SETUP OUTPUT FOLDER -------------------------
os.makedirs("my_output_small", exist_ok=True)
LOG_FILE = os.path.join("my_output_small", "pipeline_log_small.txt")

def log(msg):
    """Append to 'pipeline_log_small.txt' instead of printing."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

# ------------------------- 2. LOAD & SUBSAMPLE DATA -------------------------
df = pd.read_csv("hiring_migration_bias.csv")

# Sub-sample 20 random rows
df_small = df.sample(n=20, random_state=42).copy()

# Binary protected attribute: 1 if gender=1, else 0
df_small["is_woman"] = df_small["Gender"].apply(lambda x: 1 if x == 1 else 0)

# Define columns
FEATURES = [
    "Country",
    "Age_Group",
    "Age_Group_Label",
    "Education_Level",
    "Professional_Developer",
    "YearsCode",
    "Employment",
    "Pct_Female_HigherEd",
    "Pct_Male_HigherEd",
    "Pct_Female_MidEd",
    "Pct_Male_MidEd",
    "Pct_Female_LowEd",
    "Pct_Male_LowEd"
]

NUMERIC_COLS = [
    "Age_Group",
    "Education_Level",
    "Professional_Developer",
    "YearsCode",
    "Employment",
    "Pct_Female_HigherEd",
    "Pct_Male_HigherEd",
    "Pct_Female_MidEd",
    "Pct_Male_MidEd",
    "Pct_Female_LowEd",
    "Pct_Male_LowEd"
]
CAT_COLS = ["Country", "Age_Group_Label"]

X_small = df_small[FEATURES].copy()
y_small = df_small["Employed"].copy()
prot_small = df_small["is_woman"].copy()

# ------------------------- 3. OUTLIER CLIPPING (NUMPY-BASED) -------------------------
def clip_outliers_array(X_in: np.ndarray):
    """
    Clip each column to [0.5th, 99.5th percentile].
    On 20 rows, effect is minimal, but keeps code consistent.
    """
    X_out = np.copy(X_in)
    for col_idx in range(X_out.shape[1]):
        col_data = X_out[:, col_idx]
        low_q = np.percentile(col_data, 0.5)
        high_q = np.percentile(col_data, 99.5)
        X_out[:, col_idx] = np.clip(col_data, low_q, high_q)
    return X_out

clipper = FunctionTransformer(clip_outliers_array)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("clipper", clipper),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_COLS),
    ("cat", categorical_transformer, CAT_COLS)
])

# ------------------------- 4. TRAIN/TEST SPLIT (SMALL) -------------------------
X_train_s, X_test_s, y_train_s, y_test_s, prot_train_s, prot_test_s = train_test_split(
    X_small, y_small, prot_small,
    test_size=0.2,
    random_state=42,
    stratify=y_small if len(y_small.unique()) > 1 else None
)

# ------------------------- 5. XGBOOST PIPELINE -------------------------
pipe_small = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBClassifier(eval_metric="logloss", use_label_encoder=False))
])

# ------------------------- 6. TINY HYPERPARAMETER SEARCH -------------------------
param_distributions_small = {
    "xgb__n_estimators": [50, 100],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.1, 0.2],
}

cv_small = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

search_small = RandomizedSearchCV(
    estimator=pipe_small,
    param_distributions=param_distributions_small,
    n_iter=2,          # extremely small for speed
    scoring="f1",
    cv=cv_small,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search_small.fit(X_train_s, y_train_s)
best_small_model = search_small.best_estimator_

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("BEST PARAMS (small run):\n" + str(search_small.best_params_) + "\n")

# ------------------------- 7. BASELINE PREDICTIONS & METRICS -------------------------
y_pred_s = best_small_model.predict(X_test_s)
acc_s = accuracy_score(y_test_s, y_pred_s)
prec_s = precision_score(y_test_s, y_pred_s, zero_division=0)
rec_s = recall_score(y_test_s, y_pred_s, zero_division=0)
f1_s = f1_score(y_test_s, y_pred_s, zero_division=0)

log("\n==== Baseline Model (small subset) ====")
log(f"Accuracy:  {acc_s:.4f}")
log(f"Precision: {prec_s:.4f}")
log(f"Recall:    {rec_s:.4f}")
log(f"F1:        {f1_s:.4f}")

mf_small_base = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score
    },
    y_true=y_test_s,
    y_pred=y_pred_s,
    sensitive_features=prot_test_s
)
mf_small_base.by_group.to_csv(os.path.join("my_output_small", "baseline_by_group_small.csv"))

dp_diff_s = demographic_parity_difference(y_test_s, y_pred_s, sensitive_features=prot_test_s)
dp_ratio_s = demographic_parity_ratio(y_test_s, y_pred_s, sensitive_features=prot_test_s)
eo_diff_s = equalized_odds_difference(y_test_s, y_pred_s, sensitive_features=prot_test_s)
eo_ratio_s = equalized_odds_ratio(y_test_s, y_pred_s, sensitive_features=prot_test_s)

log("\n==== Fairness Metrics (small baseline) ====")
log(f"Demographic Parity Difference: {dp_diff_s:.4f}")
log(f"Demographic Parity Ratio:      {dp_ratio_s:.4f}")
log(f"Equalized Odds Difference:     {eo_diff_s:.4f}")
log(f"Equalized Odds Ratio:          {eo_ratio_s:.4f}")

# ------------------------- 8. IN-PROCESSING FAIRNESS -------------------------
# Must direct sample_weight to 'xgb__sample_weight'
constraint_demo = DemographicParity()

mitigator_small = ExponentiatedGradient(
    estimator=best_small_model,
    constraints=constraint_demo,
    sample_weight_name="xgb__sample_weight"
)

mitigator_small.fit(X_train_s, y_train_s, sensitive_features=prot_train_s)

# NO "sensitive_features" argument in predict for older Fairlearn versions
y_pred_mitig_s = mitigator_small.predict(X_test_s)

acc_m_s = accuracy_score(y_test_s, y_pred_mitig_s)
prec_m_s = precision_score(y_test_s, y_pred_mitig_s, zero_division=0)
rec_m_s = recall_score(y_test_s, y_pred_mitig_s, zero_division=0)
f1_m_s = f1_score(y_test_s, y_pred_mitig_s, zero_division=0)

log("\n==== Mitigated Model (small subset) ====")
log(f"Accuracy:  {acc_m_s:.4f}")
log(f"Precision: {prec_m_s:.4f}")
log(f"Recall:    {rec_m_s:.4f}")
log(f"F1:        {f1_m_s:.4f}")

dp_diff_m_s = demographic_parity_difference(y_test_s, y_pred_mitig_s, sensitive_features=prot_test_s)
dp_ratio_m_s = demographic_parity_ratio(y_test_s, y_pred_mitig_s, sensitive_features=prot_test_s)
eo_diff_m_s = equalized_odds_difference(y_test_s, y_pred_mitig_s, sensitive_features=prot_test_s)
eo_ratio_m_s = equalized_odds_ratio(y_test_s, y_pred_mitig_s, sensitive_features=prot_test_s)

log("\n==== Fairness Metrics (small mitigated) ====")
log(f"Demographic Parity Difference: {dp_diff_m_s:.4f}")
log(f"Demographic Parity Ratio:      {dp_ratio_m_s:.4f}")
log(f"Equalized Odds Difference:     {eo_diff_m_s:.4f}")
log(f"Equalized Odds Ratio:          {eo_ratio_m_s:.4f}")

# ------------------------- 9. SAVE MODELS (small) -------------------------
joblib.dump(best_small_model, os.path.join("my_output_small", "best_model_small.pkl"))
joblib.dump(mitigator_small, os.path.join("my_output_small", "mitigated_model_small.pkl"))

# ------------------------- 10. SHAP EXPLANATIONS (small) -------------------------
final_preprocessor_s = best_small_model.named_steps["preprocessor"]
final_xgb_s = best_small_model.named_steps["xgb"]

X_test_s_transformed = final_preprocessor_s.transform(X_test_s)

explainer_s = shap.TreeExplainer(final_xgb_s)
shap_values_s = explainer_s.shap_values(X_test_s_transformed)

ohe_cat_names_s = final_preprocessor_s.named_transformers_["cat"].named_steps["ohe"] \
    .get_feature_names_out(CAT_COLS)
all_feature_names_s = list(ohe_cat_names_s) + NUMERIC_COLS

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_s, features=X_test_s_transformed,
                  feature_names=all_feature_names_s,
                  plot_type="bar", show=False)
plt.savefig(os.path.join("my_output_small", "shap_summary_bar_small.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_s, features=X_test_s_transformed,
                  feature_names=all_feature_names_s, show=False)
plt.savefig(os.path.join("my_output_small", "shap_summary_beeswarm_small.png"), bbox_inches="tight")
plt.close()

log("\nSMALL TEST RUN COMPLETE. Outputs in 'my_output_small/'.")
