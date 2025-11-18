import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'Transaction_ID': range(1, 101),
    'Milk': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
    'Bread': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
    'Butter': np.random.choice([0, 1], 100, p=[0.5, 0.5]),
    'Eggs': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
    'Cereal': np.random.choice([0, 1], 100, p=[0.6, 0.4])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('market_basket.csv', index=False)
print("✅ SUCCESS: 'market_basket.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Dataset
df = pd.read_csv('market_basket.csv')
basket = df.drop('Transaction_ID', axis=1)

# 2. Apply Apriori
# min_support=0.2 means itemsets must appear in at least 20% of transactions
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# 3. Generate Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

print("\n--- Top 5 Association Rules ---")
print(rules[['antecedents_str', 'consequents_str', 'confidence', 'lift']].head())

# 4. GRAPH 1: Support vs Confidence
plt.figure(figsize=(6,4))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', s=100)
plt.colorbar(label='Lift')
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Rules: Support vs Confidence")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. GRAPH 2: Item Frequency Bar Chart
plt.figure(figsize=(6,4))
basket.sum().sort_values(ascending=False).plot(kind='bar', color='orange')
plt.title("Item Frequency (Popularity)")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. GRAPH 3: Co-occurrence Heatmap
plt.figure(figsize=(6,5))
# Calculate correlation as a proxy for co-occurrence
sns.heatmap(basket.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Item Co-occurrence Correlation")
plt.tight_layout()
plt.show()