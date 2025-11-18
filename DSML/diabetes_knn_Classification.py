import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
n = 180
data = {
    'patient_id': range(1, n+1),
    'glucose_level': np.round(np.random.normal(120, 30, n).clip(50, 300), 1),
    'bmi': np.round(np.random.normal(28, 6, n).clip(15, 50), 1),
    'age': np.random.randint(18, 90, n),
    # Create label with higher chance if glucose & BMI high
    'diabetes': np.where(
        (np.random.rand(n) + (0.6*((np.random.normal(120,30,n)-100)/100)) + (0.4*((np.random.normal(28,6,n)-25)/25))) > 0.7,
        'Yes', 'No'
    )
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('diabetes_data.csv', index=False)
print("✅ 'diabetes_data.csv' generated. Columns:", list(df_gen.columns))

# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('diabetes_data.csv')

# 2. Features & target
X = df[['glucose_level','bmi','age']]
y = df['diabetes']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 5. Predict
y_pred = knn.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Diabetes Classification (KNN) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Simple scatter
plt.figure(figsize=(6,4))
colors = df['diabetes'].map({'No':'blue','Yes':'orange'})
plt.scatter(df['glucose_level'], df['bmi'], c=colors, alpha=0.7)
plt.xlabel("Glucose Level")
plt.ylabel("BMI")
plt.title("Glucose vs BMI colored by Diabetes")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
