# Credit Risk Modeling: Optimizing Loan Approval for Profitability

## The Business Problem
A financial institution needs to improve its loan approval process. The current strategy (approving loans strictly based on internal grades A-D) is safe but leaves millions of dollars in interest revenue on the table by rejecting potentially profitable borrowers. The goal is to build a machine learning model that predicts default probability and optimizes the approval threshold for maximum portfolio profit.

## Key Findings
1. **XGBoost Outperformed Linear Models:** An XGBoost gradient boosting classifier achieved an ROC-AUC of 0.6616, successfully capturing non-linear risk factors better than the baseline Logistic Regression (AUC 0.6584).
2. **Missing Data is a Behavioral Signal:** Missing employment length (`emp_length_nan`) was identified as one of the top predictors of default, indicating that null values in credit applications carry heavy risk implications.
3. **Significant Profit Margin:** Simulating a 60% Loss Given Default (LGD) on a holdout test set of 103,681 loans from 2015 revealed that optimizing the XGBoost decision threshold to 0.50 would generate ~$21.9M in profit, compared to just ~$10.2M using the legacy Grade-based rule.

## Recommendations
* **Deploy XGBoost for Underwriting:** Transition from rigid grade-based cutoffs to a dynamic, probability-based approval system using the XGBoost model.
* **Adopt a 0.50 Risk Threshold (Given Current LGD):** Based on the current 60% LGD assumption, the bank should approve loans with a predicted default probability of up to 50%, as the combined interest yield outpaces the expected principal loss. 
* **Dynamic LGD Stress Testing:** The bank's risk committee should dynamically adjust this 0.50 threshold down during macroeconomic downturns when LGD is expected to rise above 60%.

---

## Technical Execution
* **Tools Used:** Python (Pandas, Scikit-Learn, XGBoost, Matplotlib).
* **Data Integrity:** Eliminated data leakage by dropping post-origination columns (e.g., `total_pymnt`, `recoveries`) prior to model training.
* **Validation Strategy:** Utilized a strict time-based split (Train: 2013-2014, Test: 2015) rather than random splitting to mimic real-world deployment and prevent temporal leakage.
* **Class Imbalance:** Applied `scale_pos_weight` and `class_weight='balanced'` to force models to heavily penalize False Negatives (missed defaults).