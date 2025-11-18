import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# usage_km/month, monthly_bill, contract_months, churn (Yes/No)
# ==========================================
n = 300
usage = np.round(np.random.normal(300,120,n).clip(10,2000),1)
monthly_bill = np.round(200 - 0.05*usage + np.random.normal(0,20,n),2).clip(20,400)
contract_months = np.random.randint(0,36,n)
# churn more likely if low contract months & low usage & high bill
prob = 0.2 + 0.3*(contract_months<6) + 0.2*(usage<150) + 0.15*(monthly_bill>250)
labels = np.where(np.random.rand(n) < prob, 'Yes', 'No')
data = {
    'customer_id': range(1,n+1),
    'usage_min_per_month': usage,
    'monthly_bill': monthly_bill,
    'contract_months': contract_months,
    'churn': labels
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('telecom_churn.csv', index=False)
print("✅ 'telecom_churn.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('telecom_churn.csv')

# 2. Features & target
X = df[['usage_min_per_month','monthly_bill','contract_months']]
y = df['churn']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train Logistic Regression
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)

# 5. Predict & probs
y_pred = logr.predict(X_test)
y_prob = logr.predict_proba(X_test)[:,1]  # probability for positive class if encoded

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Telecom Churn (Logistic Regression) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. ROC AUC (note: need binary numeric encoding)
y_test_num = (y_test == 'Yes').astype(int)
print(f"ROC AUC (approx): {roc_auc_score(y_test_num, y_prob):.2f}")
'''
