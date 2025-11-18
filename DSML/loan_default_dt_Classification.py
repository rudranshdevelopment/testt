import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
n = 200
data = {
    'applicant_id': range(1, n+1),
    'income_k': np.round(np.random.normal(50, 20, n).clip(5, 200), 2),
    'credit_score': np.random.randint(300, 850, n),
    'years_employed': np.random.randint(0, 30, n),
    # Create a reasonable default rate dependent on low income & low credit score
    'default': np.where(
        (np.random.rand(n) + (0.5*(50 - np.round(np.random.normal(50, 20, n)).clip(5,200))/50) +
         (0.5*(700 - np.random.randint(300,850,n))/700)) > 1.2,
        'Yes', 'No'
    )
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('loan_applicants.csv', index=False)
print("✅ 'loan_applicants.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('loan_applicants.csv')

# 2. Features and target
X = df[['income_k','credit_score','years_employed']]
y = df['default']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\\n--- Loan Default (Decision Tree) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 6. Plot tree
plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=X.columns, class_names=dt.classes_, filled=True, rounded=True)
plt.title("Decision Tree for Loan Default")
plt.tight_layout()
plt.show()
'''
