import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# Simulating GPS data clusters (e.g., City Center, Highway Exit, Suburbs)
lat = np.concatenate([
    np.random.normal(18.52, 0.01, 100), # Zone A
    np.random.normal(18.55, 0.01, 100), # Zone B
    np.random.normal(18.58, 0.01, 100)  # Zone C
])
lon = np.concatenate([
    np.random.normal(73.85, 0.01, 100),
    np.random.normal(73.80, 0.01, 100),
    np.random.normal(73.90, 0.01, 100)
])

data = {'Latitude': lat, 'Longitude': lon}
df_gen = pd.DataFrame(data)
df_gen.to_csv('traffic_data.csv', index=False)
print("✅ SUCCESS: 'traffic_data.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Dataset
df = pd.read_csv('traffic_data.csv')

# 2. Features
X = df[['Latitude', 'Longitude']]

# 3. Apply K-Means (Detect 3 Congestion Zones)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Zone_Cluster'] = kmeans.fit_predict(X)

# 4. Evaluation
print("\n--- Identified Congestion Zones (Centers) ---")
print(kmeans.cluster_centers_)

# 5. GRAPH 1: Scatter Plot (Raw GPS)
plt.figure(figsize=(6,4))
plt.scatter(df['Longitude'], df['Latitude'], color='gray', alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Raw Traffic GPS Signals")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. GRAPH 2: Clustered Zones
plt.figure(figsize=(6,4))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Zone_Cluster'], cmap='cool', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
            s=200, c='black', marker='*', label='Congestion Center')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Identified Traffic Congestion Zones")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. GRAPH 3: Traffic Volume per Zone
plt.figure(figsize=(5,4))
df['Zone_Cluster'].value_counts().sort_index().plot(kind='bar', color='blue', alpha=0.6)
plt.title("Traffic Volume per Zone")
plt.xlabel("Zone ID")
plt.ylabel("Number of Signals")
plt.tight_layout()
plt.show()