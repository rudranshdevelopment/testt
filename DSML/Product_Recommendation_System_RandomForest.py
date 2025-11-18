import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'user_id': range(1, 101),
    'browsing_minutes': np.random.randint(1, 120, 100),    # Feature 1
    'past_purchases': np.random.randint(0, 50, 100),       # Feature 2
    'age': np.random.randint(18, 60, 100),                 # Feature 3
    # Target: Predicting Interest Level (High, Medium, Low)
    'interest_level': np.random.choice(['Low', 'Medium', 'High'], 100) 
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('product_recommendation.csv', index=False)
print("✅ SUCCESS: 'product_recommendation.csv' generated.") 


# ==========================================
# PART 2: SOLUTION 
# ==========================================

# 1. Load Dataset
df = pd.read_csv('product_recommendation.csv')

# 2. Features & Target
X = df[['browsing_minutes', 'past_purchases', 'age']]
y = df['interest_level']

# 3. Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predict
y_pred = rf_model.predict(X_test)

# 6. Evaluation
acc = accuracy_score(y_test, y_pred)

print("\n--- Product Recommendation Results ---")
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. GRAPH 1: Scatter Plot (Browsing vs Purchases)
plt.figure(figsize=(6,4))
plt.scatter(df['browsing_minutes'], df['past_purchases'], alpha=0.7)
plt.xlabel("Browsing Minutes")
plt.ylabel("Past Purchases Count")
plt.title("User Behavior Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. GRAPH 2: Scatter Colored by Interest Level
plt.figure(figsize=(6,4))
# Map colors
color_map = {'Low': 'red', 'Medium': 'blue', 'High': 'green'}
colors = df['interest_level'].map(color_map)
plt.scatter(df['browsing_minutes'], df['past_purchases'], c=colors, alpha=0.7)
plt.xlabel("Browsing Minutes")
plt.ylabel("Past Purchases Count")
plt.title("Interest Level by User Behavior")
# Legend
patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
plt.legend(handles=patches)
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. GRAPH 3: Accuracy Bar Chart
plt.figure(figsize=(5,4))
plt.bar(['RF Accuracy'], [acc], color='teal', alpha=0.7)
plt.ylim(0, 1.1)
plt.text(0, acc + 0.02, f"{acc:.2f}", ha='center', fontweight='bold')
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()