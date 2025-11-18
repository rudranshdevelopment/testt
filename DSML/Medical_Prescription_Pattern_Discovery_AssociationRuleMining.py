import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'Patient_ID': range(1, 101),
    'Antibiotics': np.random.choice([0, 1], 100, p=[0.5, 0.5]),
    'Painkillers': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
    'Antacids': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
    'Vitamins': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
    'Sedatives': np.random.choice([0, 1], 100, p=[0.8, 0.2])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('drug_patterns.csv', index=False)
print("✅ SUCCESS: 'drug_patterns.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('drug_patterns.csv')
drugs = df.drop('Patient_ID', axis=1)

# 2. Apriori
frequent_itemsets = apriori(drugs, min_support=0.1, use_colnames=True)

# 3. Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("\n--- Frequent Drug Combinations ---")
print(frequent_itemsets.sort_values('support', ascending=False).head())

# 4. GRAPH 1: Drug Co-Prescription Heatmap
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(drugs.corr(), annot=True, cmap="Greens", fmt=".2f")
plt.title("Drug Co-Prescription Correlations")
plt.tight_layout()
plt.show()

# 5. GRAPH 2: Top Rules Scatter
plt.figure(figsize=(6,4))
plt.scatter(rules['support'], rules['confidence'], alpha=0.6, color='red')
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Rule Strength: Support vs Confidence")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. GRAPH 3: Most Prescribed Drugs
plt.figure(figsize=(6,4))
drugs.sum().sort_values().plot(kind='barh', color='#2E8B57')
plt.title("Most Frequently Prescribed Drugs")
plt.xlabel("Count")
plt.tight_layout()
plt.show()
