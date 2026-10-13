import pandas as pd
import numpy as np

print("Loading filtered dataset...")
df = pd.read_csv('data/loan_filtered.csv', low_memory=False)

# 1. Identify and Drop Post-Origination / Future-Leakage Columns
# These columns are populated AFTER the loan is issued. If included, the model gets ~100% accuracy incorrectly.
leakage_cols = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high',
    'last_fico_range_low', 'settlement_status', 'settlement_date',
    'settlement_amount', 'settlement_percentage', 'settlement_term',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status'
]

# Drop leakage columns that exist in the dataframe
cols_to_drop = [col for col in leakage_cols if col in df.columns]
df_clean = df.drop(columns=cols_to_drop)

# 2. Select Core Features (Available AT the moment of loan application)
# We keep post-origination columns ONLY for calculating economic LGD/yield later in Phase 6, not for training!
feature_cols = [
    'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti', 'delinq_2yrs', 'fico_range_low',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'issue_year', 'target', 'grade', 'sub_grade', 'int_rate'
]

# Select existing available features
df_model = df_clean[[col for col in feature_cols if col in df_clean.columns]].copy()

# Simple imputation / cleaning
df_model['revol_util'] = df_model['revol_util'].astype(str).str.rstrip('%').astype(float)
df_model['revol_util'] = df_model['revol_util'].fillna(df_model['revol_util'].median())
df_model['annual_inc'] = df_model['annual_inc'].fillna(df_model['annual_inc'].median())
df_model['dti'] = df_model['dti'].fillna(df_model['dti'].median())

# 3. Time-Based Split: Train on 2013-2014, Test on 2015
# A time split mimics real-world deployment (predicting future loans using past history).
train_df = df_model[df_model['issue_year'].isin([2013, 2014])].copy()
test_df = df_model[df_model['issue_year'] == 2015].copy()

print("\n--- Phase 2: Train/Test Split Verification ---")
print(f"Train Set (2013-2014) Rows: {len(train_df):,} | Default Rate: {train_df['target'].mean()*100:.2f}%")
print(f"Test Set (2015) Rows:      {len(test_df):,} | Default Rate: {test_df['target'].mean()*100:.2f}%")

# Save split datasets
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print("\nTrain and test sets saved to 'data/train.csv' and 'data/test.csv'.")