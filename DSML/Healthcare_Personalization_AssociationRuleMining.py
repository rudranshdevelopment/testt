import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'Patient_ID': range(1, 101),
    'Symptom_Severity': np.random.randint(1, 10, 100),
    'Age': np.random.randint(20, 80, 100),
    'Treatment_A': np.random.choice([0, 1], 100),
    'Treatment_B': np.random.choice([0, 1], 100),
    'Treatment_C': np.random.choice([0, 1], 100)
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('health_personalization.csv', index=False)
print("✅ SUCCESS: 'health_personalization.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('health_personalization.csv')

# --- STEP A: CLUSTERING ---
X_cluster = df[['Symptom_Severity', 'Age']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# --- STEP B: ASSOCIATION RULES (on Treatments) ---
treatments = df[['Treatment_A', 'Treatment_B', 'Treatment_C']]
freq_treat = apriori(treatments, min_support=0.1, use_colnames=True)
rules = association_rules(freq_treat, metric="confidence", min_threshold=0.5)

print("--- Patient Clusters Centers ---")
print(kmeans.cluster_centers_)
print("\n--- Treatment Rules ---")
print(rules[['antecedents', 'consequents', 'confidence']].head())

# 2. GRAPH 1: Cluster Visualization
plt.figure(figsize=(6,4))
plt.scatter(df['Age'], df['Symptom_Severity'], c=df['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], c='red', marker='X', s=200)
plt.xlabel("Age")
plt.ylabel("Symptom Severity")
plt.title("Patient Segmentation (Clustering)")
plt.show()

# 3. GRAPH 2: Treatments per Cluster
plt.figure(figsize=(6,4))
df.groupby('Cluster')[['Treatment_A', 'Treatment_B', 'Treatment_C']].sum().plot(kind='bar', stacked=True)
plt.title("Treatment Distribution by Cluster")
plt.ylabel("Count")
plt.show()

# 4. GRAPH 3: Rule Confidence
plt.figure(figsize=(6,4))
if not rules.empty:
    rules['rule_name'] = rules['antecedents'].apply(list).astype(str) + "->" + rules['consequents'].apply(list).astype(str)
    plt.barh(rules['rule_name'], rules['confidence'], color='skyblue')
    plt.xlabel("Confidence")
    plt.title("Treatment Association Rules")
    plt.tight_layout()
    plt.show()
else:
    print("No strong rules found for plotting.")