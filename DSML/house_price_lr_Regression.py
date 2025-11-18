import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# Columns: location_score (numeric proxy), size_sqft, num_bedrooms, has_pool, price_k
# ==========================================
n = 220
location_score = np.round(np.random.uniform(1,10,n),2)
size_sqft = np.round(np.random.normal(1200,400,n).clip(300,8000),1)
num_bedrooms = np.random.randint(1,6,n)
has_pool = np.random.choice([0,1], n, p=[0.85,0.15])
# price roughly: base + loc*factor + size*0.1 + bedrooms*10k + pool
price_k = (50 + location_score*20 + size_sqft*0.12 + num_bedrooms*10 + has_pool*30 +
           np.random.normal(0,25,n)).round(2)
data = {
    'house_id': range(1,n+1),
    'location_score': location_score,
    'size_sqft': size_sqft,
    'num_bedrooms': num_bedrooms,
    'has_pool': has_pool,
    'price_k': price_k
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('house_prices.csv', index=False)
print("✅ 'house_prices.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('house_prices.csv')

# 2. Features & target
X = df[['location_score','size_sqft','num_bedrooms','has_pool']]
y = df['price_k']

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
print("\\n--- House Price Prediction (Linear Regression) ---")
print(f"MSE: {mse:.2f}, R2: {r2:.2f}")

# 7. Plot actual vs predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price (k)")
plt.ylabel("Predicted Price (k)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
