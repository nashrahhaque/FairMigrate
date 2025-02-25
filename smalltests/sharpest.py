import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create visuals directory explicitly
os.makedirs("baseline_results_with_bias/visuals", exist_ok=True)

# Load your dataset clearly
data = pd.read_csv("hiring_migration_bias.csv")

# Binning YearsCode explicitly
bins = [-1, 0, 2, 5, 10, 20, 50]
labels = [0, 1, 2, 3, 4, 5]
data['YearsCode_Binned'] = pd.cut(data['YearsCode'], bins=bins, labels=labels)

# Features and target explicitly
X = data[['Country', 'Age_Group', 'Education_Level', 'Professional_Developer', 
          'Employment', 'Gender', 'YearsCode_Binned',
          'Pct_Female_HigherEd', 'Pct_Male_HigherEd',
          'Pct_Female_MidEd', 'Pct_Male_MidEd', 
          'Pct_Female_LowEd', 'Pct_Male_LowEd']]
y = data['Employed']

# Encoding explicitly
X_encoded = pd.get_dummies(X, columns=['Country', 'Gender', 'YearsCode_Binned'], drop_first=True)
X_encoded = X_encoded.astype(float)  # explicitly cast to float to resolve SHAP issues

# Small subset for quick training check
X_small, _, y_small, _ = train_test_split(X_encoded, y, test_size=0.95, random_state=42)

# Train a small model explicitly for SHAP check
small_model = RandomForestClassifier(n_estimators=10, random_state=42)
small_model.fit(X_small, y_small)

# ----- Quick SHAP Check -----
explainer = shap.TreeExplainer(small_model)
shap_values = explainer.shap_values(X_small.sample(10, random_state=42), check_additivity=False)

print("SHAP calculation successful! Quick check passed.")
