import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from scipy.stats import ks_2samp

print("Loading Train and Test sets...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# 1. Define Model A features (Excluding LC's grade, sub_grade, int_rate)
target = 'target'
features = [
    'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti', 'delinq_2yrs', 'fico_range_low',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc'
]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()
y_test = test_df[target]

# 2. Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# 3. Preprocessing (Fit on Train ONLY to prevent data leakage)
# StandardScaler normalizes numbers (e.g. income vs DTI). OneHotEncoder turns text into 1s and 0s.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Logistic Regression Pipeline 
# 'balanced' forces the model to care equally about the 14% defaults and 86% good loans.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

print("Training Baseline Logistic Regression (Model A)...")
model.fit(X_train, y_train)

# 5. Predict on Test Set
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of default (Target=1)
y_pred = model.predict(X_test) # Hard prediction (0 or 1 at a 0.5 threshold)

# 6. Calculate Banking-Standard Metrics
auc = roc_auc_score(y_test, y_pred_proba)
gini = 2 * auc - 1

# KS Statistic (Kolmogorov-Smirnov) measures how well the model separates Good vs Bad borrowers
def calculate_ks(y_true, y_proba):
    preds = pd.DataFrame({'y': y_true, 'p': y_proba})
    goods = preds[preds['y'] == 0]['p']
    bads = preds[preds['y'] == 1]['p']
    return ks_2samp(goods, bads).statistic

ks_stat = calculate_ks(y_test, y_pred_proba)

print("\n--- Baseline Model Performance (Test Set) ---")
print(f"ROC-AUC: {auc:.4f} (Ideal baseline range: 0.65 - 0.70)")
print(f"Gini:    {gini:.4f}")
print(f"KS Stat: {ks_stat:.4f}")

print("\n--- Confusion Matrix (Threshold = 0.5) ---")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Good predicted Good): {cm[0][0]}")
print(f"False Positives (Good predicted Bad):  {cm[0][1]} (Lost revenue)")
print(f"False Negatives (Bad predicted Good):  {cm[1][0]} (Financial loss)")
print(f"True Positives (Bad predicted Bad):   {cm[1][1]} (Risk avoided)")

# 7. Extract Coefficients in Plain English
classifier = model.named_steps['classifier']
cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

coefficients = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': classifier.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n--- Top 5 Model Drivers ---")
print(coefficients.head(5).to_string(index=False))