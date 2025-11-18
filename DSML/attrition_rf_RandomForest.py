import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# features: employee_id, job_satisfaction(1-5), performance_score(1-5), years_at_company, attrition (Yes/No)
# ==========================================
n = 300
job_sat = np.random.randint(1,6,n)
perf = np.random.randint(1,6,n)
years = np.random.randint(0,25,n)
# attrition higher if low satisfaction and low performance and few years
prob = 0.1 + 0.2*(job_sat<=2) + 0.15*(perf<=2) + 0.1*(years<2)
labels = np.where(np.random.rand(n) < prob, 'Yes', 'No')
data = {
    'employee_id': range(1,n+1),
    'job_satisfaction': job_sat,
    'performance_score': perf,
    'years_at_company': years,
    'attrition': labels
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('employee_attrition.csv', index=False)
print("✅ 'employee_attrition.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('employee_attrition.csv')

# 2. Features & target
X = df[['job_satisfaction','performance_score','years_at_company']]
y = df['attrition']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# 4. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# 5. Predict
y_pred = rf.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Employee Attrition (Random Forest) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Feature importance
importances = rf.feature_importances_
for f, imp in zip(X.columns, importances):
    print(f"{f}: {imp:.3f}")
'''
