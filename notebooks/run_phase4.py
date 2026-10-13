import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix

print("Loading Train and Test sets...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

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

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# XGBoost uses a specific parameter to balance classes instead of 'class_weight'
# scale_pos_weight = (count of negative examples) / (count of positive examples)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight, 
        random_state=42,
        eval_metric='auc',
        max_depth=4,            # Keep trees shallow to prevent overfitting
        learning_rate=0.1,
        n_estimators=100
    ))
])

print("Training XGBoost (Model B)...")
model_xgb.fit(X_train, y_train)

y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = model_xgb.predict(X_test)

auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print("\n--- XGBoost Performance (Test Set) ---")
print(f"XGBoost ROC-AUC: {auc_xgb:.4f} (Target to beat: 0.6584)")

print("\n--- XGBoost Confusion Matrix (Threshold = 0.5) ---")
cm = confusion_matrix(y_test, y_pred_xgb)
print(f"True Negatives (Good predicted Good): {cm[0][0]}")
print(f"False Positives (Good predicted Bad):  {cm[0][1]} (Lost revenue)")
print(f"False Negatives (Bad predicted Good):  {cm[1][0]} (Financial loss)")
print(f"True Positives (Bad predicted Bad):   {cm[1][1]} (Risk avoided)")

# Save probabilities for the business layer (Phase 6)
test_df['lr_prob'] = y_test # Placeholder, we will rely on XGBoost
test_df['xgb_prob'] = y_pred_proba_xgb
test_df.to_csv('data/test_with_probs.csv', index=False)