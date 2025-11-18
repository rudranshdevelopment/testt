
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# Simulating 1000 pixels with RGB values (0-255)
data = {
    'R': np.random.randint(0, 255, 1000),
    'G': np.random.randint(0, 255, 1000),
    'B': np.random.randint(0, 255, 1000)
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('image_pixels.csv', index=False)
print("✅ SUCCESS: 'image_pixels.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Dataset
df = pd.read_csv('image_pixels.csv')

# 2. Features (R, G, B)
X = df[['R', 'G', 'B']]

# 3. Apply K-Means (Compress to 8 colors)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Get the compressed colors (centroids)
compressed_colors = kmeans.cluster_centers_.astype(int)
print("\n--- Compressed Palette (RGB Centroids) ---\n", compressed_colors)

# 4. GRAPH 1: 3D Scatter of Original Pixels (Subset for visibility)
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
# Normalize colors for matplotlib (0-1 range)
ax.scatter(df['R'], df['G'], df['B'], c=df[['R','G','B']].values/255.0, s=10)
ax.set_title("Original Pixel Colors (3D RGB)")
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
plt.show()

# 5. GRAPH 2: The Compressed Palette
plt.figure(figsize=(6,2))
plt.imshow([compressed_colors], aspect='auto')
plt.axis('off')
plt.title("Compressed Color Palette (K=8)")
plt.show()

# 6. GRAPH 3: Pixel Distribution per Color Cluster
plt.figure(figsize=(5,4))
df['Cluster'].value_counts().sort_index().plot(kind='bar', color='gray', alpha=0.7)
plt.title("Pixels per Compressed Color")
plt.xlabel("Color Cluster ID")
plt.ylabel("Pixel Count")
plt.tight_layout()
plt.show()