import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data... (This may take a minute for the massive Lending Club file)")
# low_memory=False prevents pandas from crashing on mixed data types
df_raw = pd.read_csv('data/loan.csv', low_memory=False)

# 1. Filter to 36-month loans only 
# (60-month loans from recent years haven't had time to default, causing "censoring bias")
df = df_raw[df_raw['term'].astype(str).str.contains('36', na=False)].copy()

# 2. Filter to Issue Years 2013-2015
# Handle various date string formats safely
df['issue_year'] = pd.to_datetime(df['issue_d'], format='mixed').dt.year
df = df[df['issue_year'].isin([2013, 2014, 2015])]

# 3. Keep only terminal statuses (Finished loans)
terminal_statuses = ['Fully Paid', 'Charged Off', 'Default']
df = df[df['loan_status'].isin(terminal_statuses)].copy()

# 4. Create our Target Variable (1 = Bad Loan, 0 = Good Loan)
df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# 5. Downsample for local machine performance
if len(df) > 200000:
    df = df.sample(n=200000, random_state=42)

print("\n--- Phase 1: Initial Health Check ---")
print(f"Total valid loans in sample: {len(df)}")
print(f"Overall Default Rate: {df['target'].mean() * 100:.2f}%\n")

print("--- Default Rate by LendingClub Grade ---")
grade_risk = df.groupby('grade')['target'].agg(['count', 'mean']).reset_index()
grade_risk['mean'] = grade_risk['mean'] * 100
grade_risk.rename(columns={'count': 'total_loans', 'mean': 'default_rate_pct'}, inplace=True)
print(grade_risk.to_string(index=False))

# 6. Save this clean baseline for Phase 2
df.to_csv('data/loan_filtered.csv', index=False)
print("\nSaved filtered dataset to 'data/loan_filtered.csv'")