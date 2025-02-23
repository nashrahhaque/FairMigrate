import pandas as pd
import os

# Ensure output directory exists
output_folder = "aifairness/tables"
os.makedirs(output_folder, exist_ok=True)

# Load bias analysis results
chi_square_results = pd.read_csv("aifairness/tables/chi_square_results.csv")
young_bias = pd.read_csv("aifairness/tables/young_disparate_impact.csv")
older_bias = pd.read_csv("aifairness/tables/older_disparate_impact.csv")

# Load world-level bias data (Merged Disparate Impact)
world_bias = pd.concat([young_bias, older_bias], keys=["Young", "Older"], names=["Age_Group"])
world_bias = world_bias.reset_index()

# Compute global statistics
num_countries = world_bias["Country"].nunique()
num_countries_biased = world_bias[world_bias["Disparate_Impact_Ratio"] < 1].shape[0]
num_countries_fair = world_bias[
    (world_bias["Disparate_Impact_Ratio"] >= 1.0) & (world_bias["Disparate_Impact_Ratio"] <= 1.3)
].shape[0]
num_countries_more_women_hired = world_bias[world_bias["Disparate_Impact_Ratio"] > 1.3].shape[0]

# Find most biased countries
severely_biased_countries = world_bias.sort_values(by="Disparate_Impact_Ratio").head(5)["Country"].tolist()
moderately_biased_countries = world_bias[
    (world_bias["Disparate_Impact_Ratio"] >= 0.7) & (world_bias["Disparate_Impact_Ratio"] < 1)
].head(5)["Country"].tolist()
fair_hiring_countries = world_bias[
    (world_bias["Disparate_Impact_Ratio"] >= 1.0) & (world_bias["Disparate_Impact_Ratio"] <= 1.3)
].head(5)["Country"].tolist()
more_women_hired_countries = world_bias[world_bias["Disparate_Impact_Ratio"] > 1.3].head(5)["Country"].tolist()

# Compare young vs. older bias levels
young_bias_avg = young_bias["Disparate_Impact_Ratio"].mean()
older_bias_avg = older_bias["Disparate_Impact_Ratio"].mean()
bias_stronger_in_young = "Yes" if young_bias_avg < older_bias_avg else "No"
bias_stronger_in_older = "Yes" if older_bias_avg < young_bias_avg else "No"

# Prepare Summary Table
summary_data = {
    "Analysis Type": [
        "Chi-Square Test",
        "Total Countries Analyzed",
        "Countries with Bias Against Women (<1.0)",
        "Countries with Fair Hiring (1.0 - 1.3)",
        "Countries Where More Women are Hired (>1.3)",
        "Severely Biased Countries (<0.7)",
        "Moderately Biased Countries (0.7 - 1.0)",
        "Fair Hiring Countries (1.0 - 1.3)",
        "More Women Hired Countries (>1.3)",
        "Bias Stronger in Younger Applicants (<35)",
        "Bias Stronger in Older Applicants (≥35)"
    ],
    "Key Findings": [
        f"Significant bias detected (p < 0.05)" if chi_square_results["p-value"][0] < 0.05 else "No significant bias detected",
        num_countries,
        num_countries_biased,
        num_countries_fair,
        num_countries_more_women_hired,
        ", ".join(severely_biased_countries),
        ", ".join(moderately_biased_countries),
        ", ".join(fair_hiring_countries),
        ", ".join(more_women_hired_countries),
        bias_stronger_in_young,
        bias_stronger_in_older
    ]
}

# Save the Summary Table
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_folder, "bias_analysis_summary.csv")
summary_df.to_csv(summary_path, index=False)

# Print Summary Insights
print("\n📊 **Bias Analysis Summary**")
print(summary_df)

print(f"\n✅ Bias analysis summary saved at: {summary_path}")
