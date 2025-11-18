import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'CustomerID': range(1, 201),
    'Annual_Income_k$': np.concatenate([
        np.random.normal(30, 10, 50), 
        np.random.normal(70, 15, 50), 
        np.random.normal(100, 20, 100)
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(80, 10, 50), 
        np.random.normal(50, 15, 50), 
        np.random.normal(20, 20, 100)
    ])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('customer_segmentation.csv', index=False)
print("✅ SUCCESS: 'customer_segmentation.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Dataset
df = pd.read_csv('customer_segmentation.csv')

# 2. Features
X = df[['Annual_Income_k$', 'Spending_Score']]

# 3. Apply K-Means Clustering
# We assume 5 clusters (standard for this famous dataset structure)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Evaluation (Print Centers)
print("\n--- Customer Segments (Centroids) ---")
print(kmeans.cluster_centers_)
print("\nCounts per Cluster:\n", df['Cluster'].value_counts())

# 5. GRAPH 1: Scatter Plot (Raw Data)
plt.figure(figsize=(6,4))
plt.scatter(df['Annual_Income_k$'], df['Spending_Score'], alpha=0.6)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Raw Customer Data")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. GRAPH 2: Clusters Visualization
plt.figure(figsize=(6,4))
plt.scatter(df['Annual_Income_k$'], df['Spending_Score'], c=df['Cluster'], cmap='viridis', alpha=0.6)
# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. GRAPH 3: Bar Chart of Distribution
plt.figure(figsize=(5,4))
df['Cluster'].value_counts().sort_index().plot(kind='bar', color='teal', alpha=0.7)
plt.title("Number of Customers per Segment")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.tight_layout()
plt.show()