import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# 1 indicates the user visited that page during the session
data = {
    'Session_ID': range(1, 101),
    'HomePage': np.random.choice([0, 1], 100, p=[0.1, 0.9]),
    'LoginPage': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
    'ProductPage': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
    'CartPage': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
    'CheckoutPage': np.random.choice([0, 1], 100, p=[0.7, 0.3])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('web_navigation.csv', index=False)
print("✅ SUCCESS: 'web_navigation.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('web_navigation.csv')
sessions = df.drop('Session_ID', axis=1)

# 2. Apriori
frequent_itemsets = apriori(sessions, min_support=0.3, use_colnames=True)

# 3. Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules['path'] = rules['antecedents'].apply(lambda x: list(x)[0]) + " -> " + rules['consequents'].apply(lambda x: list(x)[0])

print("\n--- Top Navigation Paths (by Lift) ---")
print(rules[['path', 'support', 'lift']].sort_values(by='lift', ascending=False).head())

# 4. GRAPH 1: Top Frequent Page Sets
plt.figure(figsize=(6,4))
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
top_sets = frequent_itemsets[frequent_itemsets['length'] > 1].sort_values('support', ascending=False).head(5)
labels = [str(list(x)) for x in top_sets['itemsets']]
plt.barh(labels, top_sets['support'], color='teal')
plt.xlabel("Support")
plt.title("Top Frequent Navigation Paths")
plt.tight_layout()
plt.show()

# 5. GRAPH 2: Rule Lift Bar Chart
plt.figure(figsize=(6,4))
top_rules = rules.sort_values('lift', ascending=False).head(5)
plt.bar(top_rules['path'], top_rules['lift'], color='purple')
plt.xticks(rotation=45, ha='right')
plt.title("Strongest Association Rules (Lift)")
plt.ylabel("Lift Score")
plt.tight_layout()
plt.show()

# 6. GRAPH 3: Page Visit Counts
plt.figure(figsize=(6,4))
sessions.sum().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel("")
plt.title("Distribution of Page Visits")
plt.tight_layout()
plt.show()