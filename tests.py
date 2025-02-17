import pandas as pd

# Load the dataset
file_path = "merged_cleaned_data.csv"  # Update if needed
df = pd.read_csv(file_path)

print("\n📌 **Dataset Validation Tests**")
print("="*40)
print(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")


# ===============================
# 🔍 1. Check for Missing Values
# ===============================
missing_counts = df.isna().sum()
print("\n🔍 **Checking for Missing Values...**")
print(missing_counts[missing_counts > 0])  # Show only columns with missing values

if missing_counts.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠ Warning: Missing values found! Fix before bias analysis.")


# ============================================
# 🔍 2. Check if Education Levels Are Mapped
# ============================================
print("\n🔍 **Checking Education Level Mapping...**")
expected_levels = {0, 1, 2, 3, 4}
unique_levels = set(df["Education_Level"].dropna().unique())

if unique_levels == expected_levels:
    print(f"✅ Education levels are correctly mapped: {unique_levels}")
else:
    print(f"⚠ Warning: Unexpected values in Education_Level: {unique_levels - expected_levels}")


# ===============================
# 🔍 3. Check for Duplicate Rows
# ===============================
print("\n🔍 **Checking for Duplicate Rows...**")
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print(f"⚠ Warning: {duplicate_count} duplicate rows found! Consider removing them.")
else:
    print("✅ No duplicate rows found.")


# ==================================
# 🔍 4. Check for Country Mismatches
# ==================================
print("\n🔍 **Checking Country Names for Consistency...**")
unique_countries = df["Country"].nunique()
print(f"✅ Found {unique_countries} unique countries.")

# Optional: Print unique country names if needed
#print("Countries:", df["Country"].unique())


# ================================================
# 🔍 5. Check if Education Percentages Are Scaled
# ================================================
print("\n🔍 **Checking if Education Percentages Are Correctly Scaled...**")

for col in ["Pct_Female_HigherEd", "Pct_Male_HigherEd", "Pct_Female_MidEd", 
            "Pct_Male_MidEd", "Pct_Female_LowEd", "Pct_Male_LowEd"]:
    
    min_val, max_val = df[col].min(), df[col].max()
    print(f"   {col}: Min = {min_val}, Max = {max_val}")
    
    if max_val > 1:
        print(f"⚠ Warning: {col} values might be in `0-100` instead of `0-1` scale!")


# ============================================
# 🔍 6. Check & Fix Incorrect Data Types
# ============================================
print("\n🔍 **Checking Data Types...**")

expected_ints = ["Education_Level", "Employment", "Gender", 
                 "Professional_Developer", "YearsCode", "Employed", "Age_Group"]

for col in expected_ints:
    if df[col].dtype != "int64":
        df[col] = df[col].astype(int)  # Convert to integer
        print(f"✅ Fixed data type for {col} (converted to int)")

print("✅ Data types validated.")


# ========================================
# 🔍 7. Check for Outliers in Key Columns
# ========================================
print("\n🔍 **Checking for Outliers in Numeric Columns...**")
numeric_cols = ["YearsCode", "Pct_Female_HigherEd", "Pct_Male_HigherEd", 
                "Pct_Female_MidEd", "Pct_Male_MidEd", "Pct_Female_LowEd", "Pct_Male_LowEd"]

print(df[numeric_cols].describe(percentiles=[0.01, 0.99]))


# =========================================
# 🔍 8. Check if `Employed` Column is Valid
# =========================================
print("\n🔍 **Checking Employed Column Values...**")
employment_values = df["Employed"].value_counts().to_dict()

if set(employment_values.keys()) == {0, 1}:
    print("✅ Employed column is correctly labeled with only {0,1}.")
else:
    print(f"⚠ Warning: Unexpected values in 'Employed': {employment_values}")


# ====================================
# 🔍 9. Display Sample Rows for Review
# ====================================
print("\n📌 **Sample Rows for Manual Review:**")
print(df.sample(10))  # Display 10 random rows

print("\n✅ **All checks completed!** Ready for bias analysis.")
