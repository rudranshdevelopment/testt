import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# features: age, cholesterol_mg_dl, resting_bp, max_heart_rate, risk (High/Low)
# ==========================================
n = 240
age = np.random.randint(30,85,n)
chol = np.round(np.random.normal(200,40,n).clip(100,400),1)
rest_bp = np.round(np.random.normal(120,15,n).clip(80,220),1)
max_hr = np.round(np.random.normal(150,20,n).clip(80,210),1)
risk_score = 0.02*age + 0.01*(chol-150) + 0.015*(rest_bp-100) - 0.02*(max_hr-130)
labels = np.where(risk_score + np.random.normal(0,1,n) > 1.5, 'High', 'Low')
data = {
    'patient_id': range(1,n+1),
    'age': age,
    'cholesterol_mg_dl': chol,
    'resting_bp': rest_bp,
    'max_heart_rate': max_hr,
    'risk': labels
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('heart_disease.csv', index=False)
print("✅ 'heart_disease.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('heart_disease.csv')

# 2. Features & target
X = df[['age','cholesterol_mg_dl','resting_bp','max_heart_rate']]
y = df['risk']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train Logistic Regression
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)

# 5. Predict
y_pred = logr.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Heart Disease Risk (Logistic Regression) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Plot distribution by risk
plt.figure(figsize=(6,4))
df['risk'].value_counts().plot(kind='bar')
plt.title("Risk distribution")
plt.tight_layout()
plt.show()
'''
