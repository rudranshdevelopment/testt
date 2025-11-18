import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# features: amount, transaction_hour, location_id, is_international, label Fraud/Legit
# ==========================================
n = 1000
amount = np.round(np.random.exponential(scale=80, size=n).clip(1,5000),2)
transaction_hour = np.random.randint(0,24,n)
location_id = np.random.randint(1,200,n)
is_international = np.random.choice([0,1], n, p=[0.95,0.05])
# fraud more likely for high amount + international + odd hours
fraud_prob = 0.02 + 0.0002*amount + 0.25*is_international + 0.1*((transaction_hour<6)|(transaction_hour>22))
labels = np.where(np.random.rand(n) < fraud_prob, 'Fraud', 'Legit')
data = {
    'tx_id': range(1,n+1),
    'amount': amount,
    'transaction_hour': transaction_hour,
    'location_id': location_id,
    'is_international': is_international,
    'label': labels
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('transactions.csv', index=False)
print("✅ 'transactions.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('transactions.csv')

# 2. Features & target
X = df[['amount','transaction_hour','location_id','is_international']]
y = df['label']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# 4. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

# 5. Predict
y_pred = rf.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Fraud Detection (Random Forest) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Legit','Fraud'])
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0,1], ['Legit','Fraud'])
plt.yticks([0,1], ['Legit','Fraud'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.tight_layout()
plt.show()
'''
