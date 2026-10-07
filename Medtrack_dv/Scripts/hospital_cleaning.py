"""
============================================================
MedTrack_DV — Module 2: Data Cleaning & Transformation
============================================================
Input  : healthcare_dataset.csv
Output : hospital_cleaned.csv

Cleaning Steps:
  Step 1 : Raw data load + initial report
  Step 2 : Column names standardize karo
  Step 3 : Name column fix karo
  Step 4 : Date format fix karo
  Step 5 : Negative billing fix karo
  Step 6 : Duplicate rows check + remove
  Step 7 : Null values check + handle
  Step 8 : Pediatrics imbalance fix karo
  Step 9 : Unnecessary columns drop karo
  Step 10: Length_of_Stay_Days add karo
  Step 11: Final verification + save
============================================================
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# STEP 1: Raw data load + initial report
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Raw data load ho raha hai...")
print("=" * 60)

df = pd.read_csv("healthcare_dataset.csv")

print(f"\n  Raw Rows    : {len(df):,}")
print(f"  Raw Columns : {len(df.columns)}")
print(f"\n  Columns     : {list(df.columns)}")
print(f"\n  Sample (3 rows):")
print(df.head(3).to_string())

print(f"\n  Null values :")
print(df.isnull().sum().to_string())

print(f"\n  Duplicates  : {df.duplicated().sum()}")
print(f"\n  Billing Amount negatives : {(df['Billing Amount'] < 0).sum()}")


# ─────────────────────────────────────────────
# STEP 2: Column names standardize karo
# Spaces hata do, snake_case mein convert karo
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Column names standardize ho rahe hain...")
print("=" * 60)

df = df.rename(columns={
    "Patient Id"       : "patient_id",
    "Name"             : "patient_name",
    "Age"              : "patient_age",
    "Gender"           : "patient_gender",
    "Blood Type"       : "blood_type",
    "Medical Condition": "medical_condition",
    "Date of Admission": "admission_date",
    "Department"       : "department",
    "Doctor"           : "doctor_name",
    "Hospital"         : "hospital_name",
    "Insurance Provider": "insurance_provider",
    "Hospital_Region"  : "hospital_region",
    "Billing Amount"   : "billing_amount",
    "Room Number"      : "room_number",
    "Admission_Type"   : "admission_type",
    "Discharge Date"   : "discharge_date",
    "Medication"       : "medication",
    "Test Results"     : "test_results",
})

print(f"  ✓ Columns renamed: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 3: Name column fix karo
# "Bobby JacksOn" → "Bobby Jackson"
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: patient_name column fix ho raha hai...")
print("=" * 60)

print(f"\n  Before (5 samples):")
print(df['patient_name'].head(5).to_string())

df['patient_name'] = df['patient_name'].str.strip().str.title()

print(f"\n  After (5 samples):")
print(df['patient_name'].head(5).to_string())
print(f"\n  ✓ patient_name fixed — proper case applied")


# ─────────────────────────────────────────────
# STEP 4: Date format fix karo
# "31-01-2024" (DD-MM-YYYY) → "2024-01-31" (YYYY-MM-DD)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Date format fix ho raha hai...")
print("=" * 60)

print(f"\n  Before admission_date (5): {df['admission_date'].head().tolist()}")
print(f"  Before discharge_date (5): {df['discharge_date'].head().tolist()}")

df['admission_date'] = pd.to_datetime(
    df['admission_date'], format='%d-%m-%Y', dayfirst=True
)
df['discharge_date'] = pd.to_datetime(
    df['discharge_date'], format='%d-%m-%Y', dayfirst=True
)

# Discharge date admission se pehle to nahi hai check karo
invalid_dates = (df['discharge_date'] < df['admission_date']).sum()
print(f"\n  Discharge < Admission (errors): {invalid_dates}")

if invalid_dates > 0:
    mask = df['discharge_date'] < df['admission_date']
    df.loc[mask, 'discharge_date'] = df.loc[mask, 'admission_date'] + pd.Timedelta(days=1)
    print(f"  ✓ {invalid_dates} date errors fixed")

print(f"\n  After admission_date (5): {df['admission_date'].dt.strftime('%Y-%m-%d').head().tolist()}")
print(f"  After discharge_date (5): {df['discharge_date'].dt.strftime('%Y-%m-%d').head().tolist()}")
print(f"  ✓ Dates converted to YYYY-MM-DD format")


# ─────────────────────────────────────────────
# STEP 5: Negative billing amount fix karo
# Negative values = data entry error
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Negative billing amount fix ho raha hai...")
print("=" * 60)

neg_count = (df['billing_amount'] < 0).sum()
print(f"\n  Negative values found : {neg_count}")
print(f"  Min before fix        : {df['billing_amount'].min():.2f}")

df['billing_amount'] = df['billing_amount'].abs()

print(f"  Min after fix         : {df['billing_amount'].min():.2f}")
print(f"  Max                   : {df['billing_amount'].max():.2f}")
print(f"  Avg                   : {df['billing_amount'].mean():.2f}")
print(f"  ✓ {neg_count} negative values → positive kiye (abs value)")


# ─────────────────────────────────────────────
# STEP 6: Duplicate rows check + remove
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Duplicate rows check ho rahi hain...")
print("=" * 60)

before = len(df)
dup_count = df.duplicated().sum()
print(f"\n  Duplicates found : {dup_count}")

df = df.drop_duplicates()

after = len(df)
print(f"  Rows before      : {before:,}")
print(f"  Rows after       : {after:,}")
print(f"  Removed          : {before - after}")
print(f"  ✓ Duplicates handled")


# ─────────────────────────────────────────────
# STEP 7: Null values check + handle
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Null values check ho rahi hain...")
print("=" * 60)

null_counts = df.isnull().sum()
total_nulls  = null_counts.sum()

print(f"\n  Total null values : {total_nulls}")

if total_nulls == 0:
    print(f"  ✓ No null values — dataset clean hai")
else:
    print(f"\n  Nulls per column:")
    print(null_counts[null_counts > 0].to_string())

    # Numeric columns — median se fill karo
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            print(f"  ✓ {col} → filled with median ({median_val:.1f})")

    # Categorical columns — mode se fill karo
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            print(f"  ✓ {col} → filled with mode ({mode_val})")

print(f"  Remaining nulls : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 8: Pediatrics imbalance fix karo
# Sirf 116 rows hain — baaki departments mein thousands
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Department distribution check + fix...")
print("=" * 60)

print(f"\n  Department counts before:")
print(df['department'].value_counts().to_string())

ped_count = (df['department'] == 'Pediatrics').sum()
print(f"\n  Pediatrics rows : {ped_count} — too low!")

# Age < 18 wale rows ko Pediatrics assign karo
# Yeh realistic approach hai
age_under_18 = (df['patient_age'] < 18)
df.loc[age_under_18, 'department'] = 'Pediatrics'

new_ped = (df['department'] == 'Pediatrics').sum()
print(f"  Pediatrics after fix : {new_ped}")
print(f"\n  Department counts after:")
print(df['department'].value_counts().to_string())
print(f"  ✓ Pediatrics balanced — age < 18 → Pediatrics mapped")


# ─────────────────────────────────────────────
# STEP 9: Unnecessary columns drop karo
# Project requirements mein kaam nahi aayenge
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Unnecessary columns drop ho rahe hain...")
print("=" * 60)

drop_cols = [
    'patient_name',      # PII — dashboard mein use nahi hoga
    'doctor_name',       # Dashboard mein needed nahi
    'room_number',       # KPI ya dashboard mein use nahi
    'medication',        # Project scope se bahar
    'insurance_provider' # Dashboard mein needed nahi
]

print(f"\n  Dropping : {drop_cols}")
df = df.drop(columns=drop_cols)

print(f"  ✓ Remaining columns : {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 10: Length_of_Stay_Days add karo
# Discharge Date - Admission Date = LOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Length_of_Stay_Days calculate ho raha hai...")
print("=" * 60)

df['length_of_stay_days'] = (
    df['discharge_date'] - df['admission_date']
).dt.days

# 0 ya negative values fix karo
invalid_los = (df['length_of_stay_days'] <= 0).sum()
if invalid_los > 0:
    df.loc[df['length_of_stay_days'] <= 0, 'length_of_stay_days'] = 1
    print(f"  Fixed {invalid_los} invalid LOS values → set to 1")

print(f"\n  LOS Stats:")
print(f"  Min LOS : {df['length_of_stay_days'].min()} days")
print(f"  Max LOS : {df['length_of_stay_days'].max()} days")
print(f"  Avg LOS : {df['length_of_stay_days'].mean():.1f} days")
print(f"  ✓ length_of_stay_days column added")


# ─────────────────────────────────────────────
# STEP 11: Date helper columns add karo
# Month, Year, Quarter → Tableau trends ke liye
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Date helper columns add ho rahe hain...")
print("=" * 60)

df['admission_month']      = df['admission_date'].dt.month
df['admission_month_name'] = df['admission_date'].dt.strftime('%b')
df['admission_year']       = df['admission_date'].dt.year
df['admission_quarter']    = df['admission_date'].dt.quarter

# Dates ko string mein convert karo — Tableau friendly format
df['admission_date']  = df['admission_date'].dt.strftime('%Y-%m-%d')
df['discharge_date']  = df['discharge_date'].dt.strftime('%Y-%m-%d')

print(f"  ✓ admission_month, admission_month_name, admission_year, admission_quarter added")


# ─────────────────────────────────────────────
# FINAL: Verify + Save
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL: Verification + Save")
print("=" * 60)

print(f"\n  Final shape    : {df.shape}")
print(f"  Final columns  : {list(df.columns)}")
print(f"  Null values    : {df.isnull().sum().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")
print(f"  Billing negatives : {(df['billing_amount'] < 0).sum()}")

print(f"\n  Department distribution:")
print(df['department'].value_counts().to_string())

print(f"\n  Admission Type:")
print(df['admission_type'].value_counts().to_string())

print(f"\n  Sample (3 rows):")
print(df.head(3).to_string())

# Save
df.to_csv("hospital_cleaned.csv", index=False)

print(f"\n{'=' * 60}")
print(f"  CLEANING COMPLETE!")
print(f"{'=' * 60}")
print(f"  Input  : healthcare_dataset.csv   ({55500:,} rows, 18 cols)")
print(f"  Output : hospital_cleaned.csv     ({len(df):,} rows, {len(df.columns)} cols)")
print(f"\n  Issues Fixed:")
print(f"  ✓ Column names → snake_case")
print(f"  ✓ patient_name → proper capitalization")
print(f"  ✓ Dates → YYYY-MM-DD format")
print(f"  ✓ Negative billing → absolute values")
print(f"  ✓ Pediatrics → age-based mapping")
print(f"  ✓ Unnecessary columns → dropped (5 cols)")
print(f"  ✓ length_of_stay_days → calculated from dates")
print(f"  ✓ Month, Year, Quarter → added for Tableau")
print(f"\n  Next step: Module 3 — KPI Engineering")
print(f"  Run: python generate_hospital_kpis.py")
