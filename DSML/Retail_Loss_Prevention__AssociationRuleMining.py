import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
# Simulating transactions where specific combos might be flagged
data = {
    'Trans_ID': range(1, 201),
    'HighValueItem': np.random.choice([0, 1], 200, p=[0.8, 0.2]),
    'NoReceiptReturn': np.random.choice([0, 1], 200, p=[0.9, 0.1]),
    'CashPayment': np.random.choice([0, 1], 200, p=[0.6, 0.4]),
    'LateNight': np.random.choice([0, 1], 200, p=[0.8, 0.2]),
    'Suspicious_Flag': np.random.choice([0, 1], 200, p=[0.9, 0.1]) # Target-like event
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('loss_prevention.csv', index=False)
print("✅ SUCCESS: 'loss_prevention.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('loss_prevention.csv')
trans = df.drop('Trans_ID', axis=1)

# 2. Apriori
# We look for patterns that include 'Suspicious_Flag'
freq_sets = apriori(trans, min_support=0.05, use_colnames=True)

# 3. Rules
rules = association_rules(freq_sets, metric="lift", min_threshold=1.0)

# Filter for rules leading to Suspicious_Flag
suspicious_rules = rules[rules['consequents'] == {'Suspicious_Flag'}]

print("\n--- Rules Leading to Suspicious Flag ---")
print(suspicious_rules[['antecedents', 'confidence', 'lift']].head())

# 4. GRAPH 1: Suspicious Factors Count
plt.figure(figsize=(6,4))
df[df['Suspicious_Flag'] == 1].sum().drop(['Trans_ID', 'Suspicious_Flag']).plot(kind='bar', color='maroon')
plt.title("Common Factors in Suspicious Transactions")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. GRAPH 2: Scatter of Suspicious Rules
plt.figure(figsize=(6,4))
if not suspicious_rules.empty:
    plt.scatter(suspicious_rules['support'], suspicious_rules['confidence'], s=100, c='red', label='Suspicious Rules')
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Confidence of Suspicious Patterns")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No strong rules found > 1.0 Lift", ha='center')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. GRAPH 3: Proportion of Suspicious Transactions
plt.figure(figsize=(5,5))
df['Suspicious_Flag'].value_counts().plot(kind='pie', labels=['Normal', 'Suspicious'], colors=['lightgrey', 'red'], autopct='%1.1f%%')
plt.title("Transaction Types")
plt.ylabel("")
plt.tight_layout()
plt.show()