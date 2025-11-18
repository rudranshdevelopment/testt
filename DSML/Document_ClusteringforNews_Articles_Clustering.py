import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# Simulating text data for 3 topics: Sports, Politics, Tech
topics = {
    'Sports': ['game match win score ball team player stadium goal run'],
    'Politics': ['vote election government law president senate policy minister tax'],
    'Tech': ['software ai computer algorithm data code python app internet mobile']
}

data = []
for _ in range(150): # Generate 150 articles
    category = np.random.choice(['Sports', 'Politics', 'Tech'])
    # Create a random sentence from the word bag
    words = topics[category][0].split()
    sentence = " ".join(np.random.choice(words, size=np.random.randint(5, 10)))
    data.append([sentence, category])

df_gen = pd.DataFrame(data, columns=['Article_Content', 'True_Label'])
df_gen.to_csv('news_articles.csv', index=False)
print("✅ SUCCESS: 'news_articles.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Dataset
df = pd.read_csv('news_articles.csv')

# 2. Vectorize Text (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Article_Content'])

# 3. Apply K-Means (K=3 for 3 topics)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Reduce Dimensions for Visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

print("\n--- Document Clustering Results ---")
print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(3):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :5]:
        print(f"{terms[ind]} ", end="")
    print()

# 5. GRAPH 1: Scatter Plot of Clusters (PCA Reduced)
plt.figure(figsize=(6,4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='plasma', alpha=0.7)
plt.title("Document Clusters (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. GRAPH 2: Cluster Size Distribution
plt.figure(figsize=(5,4))
df['Cluster'].value_counts().sort_index().plot(kind='bar', color='orange', alpha=0.7)
plt.title("Articles per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.tight_layout()
plt.show()