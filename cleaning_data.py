import pandas as pd
import numpy as np

# =====================================================
# 📌 Step 1: Load Datasets
# =====================================================
print("\n📌 Loading datasets...\n")

hiring_file_path = "hiring_data.csv"
immigration_file_path = "immigration_data_cleaned.csv"

# Load datasets
hiring_df = pd.read_csv(hiring_file_path)
immigration_df = pd.read_csv(immigration_file_path)

print(f"✅ Hiring dataset loaded with {hiring_df.shape[0]} rows and {hiring_df.shape[1]} columns.")
print(f"✅ Immigration dataset loaded with {immigration_df.shape[0]} rows and {immigration_df.shape[1]} columns.")
 
# =====================================================
# 📌 Step 2: Standardize Country Names & Fix Mismatches
# =====================================================
print("\n📌 Step 2: Standardizing Country Names and Fixing Mismatches...\n")

# Lowercase and trim whitespace
hiring_df["Country"] = hiring_df["Country"].astype(str).str.lower().str.strip()
immigration_df["Country"] = immigration_df["Country"].astype(str).str.lower().str.strip()

# Fix common mismatches and remove non-country entries
country_fix_map = {
    "congo, republic of the...": "congo, rep. of the",
    "côte d'ivoire": "cote d'ivoire",
    "democratic republic of the congo": "congo, dem. rep. of the",
    "gambia": "gambia, the",
    "hong kong (s.a.r.)": "china, hong kong sar",
    "iran, islamic republic of...": "iran",
    "isle of man": "united kingdom",
    "kosovo": "serbia and montenegro",
    "lao people's democratic republic": "laos",
    "libyan arab jamahiriya": "libya",
    "montenegro": "serbia and montenegro",
    "myanmar": "burma (myanmar)",
    "palestine": "occupied palestinian territory",
    "republic of korea": "korea",
    "republic of moldova": "moldova",
    "russian federation": "russia",
    "serbia": "serbia and montenegro",
    "south korea": "korea",
    "syrian arab republic": "syria",
    "the former yugoslav republic of macedonia": "macedonia",
    "timor-leste": "east timor",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "united republic of tanzania": "tanzania",
    "united states of america": "united states",
    "venezuela, bolivarian republic of...": "venezuela",
    "viet nam": "vietnam",
    # Remove non-country values (set them to None)
    "nomadic": None,
    "country of origin (mix)": None,
    "country of residence": None,
    "hum: table 1": None,
    "oecd: table 1": None,
    "total immigration stock": None,
    "total oecd": None,
    "unknown": None,
    "residents' labor force": None,
}
hiring_df["Country"] = hiring_df["Country"].replace(country_fix_map)
hiring_df = hiring_df[hiring_df["Country"].notna()]

print("✅ Country names standardized and non-country entries removed.")

# =====================================================
# 📌 Step 3: Cleaning Hiring Dataset – Age, Education, Developer Status & Target Variable
# =====================================================
print("\n📌 Step 3: Cleaning Hiring Dataset...\n")

# ----- Process Age Column -----
# Desired:
#   AI‑ready: Numeric codes (0 for Young, 1 for Older)
#   Human‑readable: "Young (<35)" and "Older (≥35)" in a column called "Age_Group"
if hiring_df["Age"].dtype == object:
    # If Age is a string (e.g., "<35" or ">35"), map it to numeric codes.
    age_str_to_code = {
        "<35": 0,
        "young (<35)": 0,
        ">35": 1,
        "older (≥35)": 1
    }
    hiring_df["Age_Code"] = hiring_df["Age"].str.strip().map(lambda x: age_str_to_code.get(x.lower(), np.nan))
else:
    hiring_df["Age_Code"] = hiring_df["Age"]

hiring_df["Age_Group"] = hiring_df["Age_Code"].map({0: "Young (<35)", 1: "Older (≥35)"})

# ----- Process Education Level -----
# Desired:
#   AI‑ready: Numeric codes 0 (Other), 1 (Undergraduate), 2 (Master), 3 (PhD)
#   Human‑readable: "Other", "Undergraduate", "Master", "PhD"
if hiring_df["EdLevel"].dtype == object:
    education_str_to_code = {
        "other": 0,
        "undergraduate": 1,
        "master": 2,
        "phd": 3
    }
    hiring_df["Education_Level_Code"] = hiring_df["EdLevel"].str.strip().str.lower().map(lambda x: education_str_to_code.get(x, np.nan))
else:
    hiring_df["Education_Level_Code"] = hiring_df["EdLevel"]

hiring_df["Education_Level"] = hiring_df["Education_Level_Code"].map({0: "Other", 1: "Undergraduate", 2: "Master", 3: "PhD"})

# ----- Process Main Branch (Developer Status) -----
# Desired:
#   AI‑ready: 0 (Not Developer) and 1 (Developer)
#   Human‑readable: "No" for 0 and "Yes" for 1 in a column "Professional_Developer"
if hiring_df["MainBranch"].dtype == object:
    dev_str_to_code = {
        "notdev": 0,
        "dev": 1
    }
    hiring_df["Professional_Developer_Code"] = hiring_df["MainBranch"].str.strip().str.lower().map(lambda x: dev_str_to_code.get(x, np.nan))
else:
    hiring_df["Professional_Developer_Code"] = hiring_df["MainBranch"]

hiring_df["Professional_Developer"] = hiring_df["Professional_Developer_Code"].map({0: "No", 1: "Yes"})

# ----- Process Gender Column -----
# Convert Gender to numeric: 0 = Male, 1 = Female, 2 = NonBinary/Other
gender_map = {"man": 0, "woman": 1}
hiring_df["Gender"] = hiring_df["Gender"].astype(str).str.strip().str.lower().map(gender_map).fillna(2)

# ----- Ensure Target Variable 'Employed' is Present -----
# This is our target variable for bias analysis.
if "Employed" in hiring_df.columns:
    hiring_df["Employed"] = hiring_df["Employed"].astype(int)
    print("✅ 'Employed' column (target variable) is correctly formatted.")
else:
    print("⚠ ERROR: 'Employed' column is missing!")

# ----- Drop Irrelevant or Redundant Columns -----
# Drop columns not needed for analysis. In this case, we remove:
# - Irrelevant columns: "HaveWorkedWith", "Accessibility"
# - Original columns that have been replaced: "Age", "EdLevel", "MainBranch"
columns_to_drop = ["HaveWorkedWith", "Accessibility", "Age", "EdLevel", "MainBranch", "PreviousSalary", "MentalHealth", "ComputerSkills"]
hiring_df.drop(columns=[col for col in columns_to_drop if col in hiring_df.columns], inplace=True)

print("✅ Hiring dataset cleaned and formatted.")

# =====================================================
# 📌 Step 4: Rename & Clean Immigration Dataset Columns
# =====================================================
print("\n📌 Step 4: Renaming Immigration Data Columns and Scaling Education Levels...\n")

# Rename columns for clarity
rename_columns = {
    "F - Ter": "Pct_Female_HigherEd",
    "M - Ter": "Pct_Male_HigherEd",
    "F - Sec": "Pct_Female_MidEd",
    "M - Sec": "Pct_Male_MidEd",
    "F - Prim": "Pct_Female_LowEd",
    "M - Prim": "Pct_Male_LowEd"
}
immigration_df.rename(columns=rename_columns, inplace=True)

# Convert "Total" column to numeric to avoid errors
immigration_df["Total"] = pd.to_numeric(immigration_df["Total"], errors="coerce").fillna(1)

# **Check if education values are already percentages**
is_percentage_data = immigration_df[rename_columns.values()].max().max() <= 100

if not is_percentage_data:
    print("⚠ Education values are raw counts, converting them to percentages...")
    # Convert absolute counts to percentages
    for col in rename_columns.values():
        immigration_df[col] = (immigration_df[col] / immigration_df["Total"]) * 100
else:
    print("✅ Education values are already percentages.")

# Ensure no values exceed 100% (clipping extreme cases)
for col in rename_columns.values():
    immigration_df[col] = immigration_df[col].clip(0, 100).round(3)

print("✅ Education level percentages have been correctly scaled and rounded.")


# =====================================================
# 📌 Step 5: Merge Hiring & Immigration Datasets
# =====================================================
print("\n📌 Step 5: Merging Hiring & Immigration Datasets...\n")

merged_df = hiring_df.merge(immigration_df, on="Country", how="left")
print(f"✅ Merged dataset created with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

if "Employed" in merged_df.columns:
    print("✅ 'Employed' target variable is present in the merged dataset.")
else:
    print("⚠ ERROR: 'Employed' target variable is missing in the merged dataset!")

print("\n🔍 Missing values BEFORE filling:")
print(merged_df.isnull().sum())

# Fill any remaining missing values with 0
merged_df.fillna(0, inplace=True)

print("\n🔍 Missing values AFTER filling:")
print(merged_df.isnull().sum())

print("\n📌 Preview of Final Merged Dataset:")
print(merged_df.head(20))

# =====================================================
# 📌 Step 6: Save Final Merged Dataset
# =====================================================
merged_data_path = "hiring_immigration_merged.csv"
merged_df.to_csv(merged_data_path, index=False)
print("\n✅ Final dataset saved:", merged_data_path)
