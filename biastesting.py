import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create directories
os.makedirs('baseline_results_with_bias', exist_ok=True)

# Load Data
data = pd.read_csv("hiring_migration_bias.csv")

# Binning YearsCode clearly
bins = [-1, 0, 2, 5, 10, 20, 50]
labels = [0, 1, 2, 3, 4, 5]
data['YearsCode_Binned'] = pd.cut(data['YearsCode'], bins=bins, labels=labels)

# Selecting features explicitly
X = data[['Country', 'Age_Group', 'Education_Level', 'Professional_Developer', 
          'Employment', 'Gender', 'YearsCode_Binned',
          'Pct_Female_HigherEd', 'Pct_Male_HigherEd',
          'Pct_Female_MidEd', 'Pct_Male_MidEd', 
          'Pct_Female_LowEd', 'Pct_Male_LowEd']]
y = data['Employed']

# Encoding categorical variables explicitly
X_encoded = pd.get_dummies(X, columns=['Country', 'Gender', 'YearsCode_Binned'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Model training explicitly
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Baseline evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Save baseline results explicitly
with open('baseline_results_with_bias/bias_analysis_results.txt', 'w') as f:
    f.write(f"Baseline Accuracy: {accuracy:.2%}\n")
    f.write("Classification Report:\n")
    f.write(classification_rep + "\n")

# Explicit bias check: Migrant Women vs. Migrant Men
# Example clearly holding constant qualifications:
def explicit_gender_bias_test(country, edu_level, years_binned):
    test_cases = pd.DataFrame({
        'Country': [country, country],
        'Age_Group': [0, 0],
        'Education_Level': [edu_level, edu_level],
        'Professional_Developer': [1, 1],
        'Employment': [1, 1],
        'Gender': [1, 0],  # Female vs Male
        'YearsCode_Binned': [years_binned, years_binned],
        'Pct_Female_HigherEd': data[data['Country'] == country]['Pct_Female_HigherEd'].mean(),
        'Pct_Male_HigherEd': data[data['Country'] == country]['Pct_Male_HigherEd'].mean(),
        'Pct_Female_MidEd': data[data['Country'] == country]['Pct_Female_MidEd'].mean(),
        'Pct_Male_MidEd': data[data['Country'] == country]['Pct_Male_MidEd'].mean(),
        'Pct_Female_LowEd': data[data['Country'] == country]['Pct_Female_LowEd'].mean(),
        'Pct_Male_LowEd': data[data['Country'] == country]['Pct_Male_LowEd'].mean()
    })
    test_cases_encoded = pd.get_dummies(test_cases, columns=['Country', 'Gender', 'YearsCode_Binned'], drop_first=True)

    missing_cols = set(X_encoded.columns) - set(test_cases_encoded.columns)
    for c in missing_cols:
        test_cases_encoded[c] = 0
    test_cases_encoded = test_cases_encoded[X_encoded.columns]

    preds = model.predict_proba(test_cases_encoded)[:, 1]
    return preds[0] - preds[1]  # Female - Male

# Conduct explicit gender bias test for a sample country
country_list = data['Country'].unique()
with open('baseline_results_with_bias/bias_analysis_results.txt', 'a') as f:
    f.write("\nExplicit Gender Bias Analysis:\n")
    for country in country_list:
        bias = explicit_gender_bias_test(country, edu_level=2, years_binned=3)
        bias_pct = bias * 100
        interpretation = f"Migrant women from {country} have {abs(bias_pct):.1f}% {'lower' if bias < 0 else 'higher'} chance of being predicted hired compared to similarly-qualified migrant men."
        f.write(interpretation + "\n")

# Country-level Bias Analysis
with open('results/bias_analysis_results.txt', 'a') as f:
    f.write("\nCountry-level Bias Analysis (women, Master's, Advanced coding exp.):\n")
    for country in country_list:
        test_case = pd.DataFrame({
            'Country': [country], 'Age_Group': [0], 'Education_Level': [2], 'Professional_Developer': [1],
            'Employment': [1], 'Gender': [1], 'YearsCode_Binned': [3],
            'Pct_Female_HigherEd': data[data['Country'] == country]['Pct_Female_HigherEd'].mean(),
            'Pct_Male_HigherEd': data[data['Country'] == country]['Pct_Male_HigherEd'].mean(),
            'Pct_Female_MidEd': data[data['Country'] == country]['Pct_Female_MidEd'].mean(),
            'Pct_Male_MidEd': data[data['Country'] == country]['Pct_Male_MidEd'].mean(),
            'Pct_Female_LowEd': data[data['Country'] == country]['Pct_Female_LowEd'].mean(),
            'Pct_Male_LowEd': data[data['Country'] == country]['Pct_Male_LowEd'].mean()
        })
        test_encoded = pd.get_dummies(test_case, columns=['Country', 'Gender', 'YearsCode_Binned'], drop_first=True)
        missing_cols = set(X_encoded.columns) - set(test_encoded.columns)
        for c in missing_cols:
            test_encoded[c] = 0
        test_encoded = test_encoded[X_encoded.columns]
        pred_prob = model.predict_proba(test_encoded)[:, 1][0]
        f.write(f"Hiring probability for migrant women from {country}: {pred_prob:.2%}\n")
