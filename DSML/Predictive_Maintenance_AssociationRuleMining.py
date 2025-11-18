import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# We ensure the data is clean by creating a fresh dictionary
data = {
    'Machine_ID': range(1, 101),
    'Temperature': np.random.randint(50, 120, 100),
    'Vibration': np.random.uniform(0.5, 5.0, 100),
    'RPM': np.random.randint(1000, 5000, 100),
    'Failure_Type': np.random.choice(['None', 'Overheat', 'PowerFailure'], 100), # Class Target
    'Days_To_Fail': np.random.randint(1, 365, 100) # Regression Target
}

df_gen = pd.DataFrame(data)
# Save to CSV (index=False prevents creating an extra unnamed column)
df_gen.to_csv('predictive_maint.csv', index=False)
print("✅ SUCCESS: 'predictive_maint.csv' generated cleanly.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('predictive_maint.csv')

# --- FIX: DROP MISSING VALUES ---
# This line fixes the "Input contains NaN" error by removing empty rows
df.dropna(inplace=True) 

# 2. Define Features and Targets
X = df[['Temperature', 'Vibration', 'RPM']]
y_class = df['Failure_Type']  # Target for Classification
y_reg = df['Days_To_Fail']    # Target for Regression

# 3. Split Data
# We split X and BOTH targets (y_class, y_reg) at the same time so indices match
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# 4. Train Models
print("Training Classification Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, yc_train)

print("Training Regression Model...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, yr_train)

# 5. Predictions & Evaluation
yc_pred = clf.predict(X_test)
yr_pred = reg.predict(X_test)

print("\n--- RESULTS ---")
print(f"Classification Accuracy: {accuracy_score(yc_test, yc_pred):.2f}")
print(f"Regression MAE (Days): {mean_absolute_error(yr_test, yr_pred):.2f}")

# 6. GRAPH 1: Scatter (Temp vs Vib by Failure Type)
plt.figure(figsize=(6,4))
colors = {'None': 'blue', 'Overheat': 'red', 'PowerFailure': 'orange'}
# We map the colors carefully. If a type isn't in the map, it defaults to gray
c_map = df['Failure_Type'].map(colors).fillna('gray')
plt.scatter(df['Temperature'], df['Vibration'], c=c_map, alpha=0.6)
plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Failure Types by Sensor Data")
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=v, label=k) for k,v in colors.items()]
plt.legend(handles=patches)
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. GRAPH 2: Regression Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(yr_test, yr_pred, color='green', alpha=0.7)
plt.plot([0, 365], [0, 365], 'k--', label='Perfect Prediction') # Ideal line
plt.xlabel("Actual Days to Fail")
plt.ylabel("Predicted Days")
plt.title("Regression: Time-to-Failure Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. GRAPH 3: Feature Importance (Classification)
plt.figure(figsize=(6,4))
plt.bar(X.columns, clf.feature_importances_, color='teal')
plt.title("Feature Importance (Failure Classification)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()