import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# rainfall_mm, avg_temp_c, soil_quality (1-10), yield_ton_per_hectare
# ==========================================
n = 180
rainfall = np.round(np.random.normal(800,150,n).clip(200,1500),1)
temp = np.round(np.random.normal(25,4,n).clip(5,45),1)
soil = np.round(np.random.uniform(3,9,n),2)
yield_t = np.round(1 + 0.002*rainfall + 0.05*(30 - abs(temp-25)) + 0.2*soil + np.random.normal(0,0.3,n),2)
data = {
    'field_id': range(1,n+1),
    'rainfall_mm': rainfall,
    'avg_temp_c': temp,
    'soil_quality': soil,
    'yield_tph': yield_t
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('crop_yield.csv', index=False)
print("✅ 'crop_yield.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('crop_yield.csv')

# 2. Features & target
X = df[['rainfall_mm','avg_temp_c','soil_quality']]
y = df['yield_tph']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5. Predict
y_pred = lr.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\\n--- Crop Yield Estimation (Linear Regression) ---")
print(f"MSE: {mse:.4f}, R2: {r2:.2f}")

# 7. Plot
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Yield (ton/ha)")
plt.ylabel("Predicted Yield (ton/ha)")
plt.title("Actual vs Predicted Crop Yield")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
