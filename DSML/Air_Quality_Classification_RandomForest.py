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
    'sensor_id': range(1, 101),
    'pm25_level': np.random.randint(10, 300, 100),     # Feature 1
    'no2_concentration': np.random.randint(5, 100, 100), # Feature 2
    'humidity': np.random.uniform(20.0, 90.0, 100),      # Feature 3
    'aqi_status': np.random.choice(['Good', 'Moderate', 'Hazardous'], 100) # Target
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('air_quality_data.csv', index=False)
print("✅ SUCCESS: 'air_quality_data.csv' generated.") 


# ==========================================
# PART 2: SOLUTION 
# ==========================================
'''
# 1. Load Dataset
df = pd.read_csv('air_quality_data.csv')

# 2. Features & Target
X = df[['pm25_level', 'no2_concentration', 'humidity']]
y = df['aqi_status']

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

print("\n--- Air Quality Classification Results ---")
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. GRAPH 1: Scatter Plot (Two Features)
plt.figure(figsize=(6,4))
plt.scatter(df['pm25_level'], df['no2_concentration'], alpha=0.7)
plt.xlabel("PM2.5 Level")
plt.ylabel("NO2 Concentration")
plt.title("Pollution Sensor Data Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. GRAPH 2: Scatter Colored by AQI Status
plt.figure(figsize=(6,4))
# Map colors to classes
color_map = {'Good': 'green', 'Moderate': 'orange', 'Hazardous': 'red'}
colors = df['aqi_status'].map(color_map)
plt.scatter(df['pm25_level'], df['no2_concentration'], c=colors, alpha=0.7)
plt.xlabel("PM2.5 Level")
plt.ylabel("NO2 Concentration")
plt.title("AQI Classification by Pollutants")
# Legend
patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
plt.legend(handles=patches)
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. GRAPH 3: Accuracy Bar Chart
plt.figure(figsize=(5,4))
plt.bar(['Random Forest Acc'], [acc], color='purple', alpha=0.7)
plt.ylim(0, 1.1)
plt.text(0, acc + 0.02, f"{acc:.2f}", ha='center', fontweight='bold')
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
'''