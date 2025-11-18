import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'Student_ID': range(1, 101),
    'Python': np.random.choice([0, 1], 100, p=[0.2, 0.8]),
    'DataScience': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
    'WebDev': np.random.choice([0, 1], 100, p=[0.5, 0.5]),
    'SQL': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
    'Java': np.random.choice([0, 1], 100, p=[0.7, 0.3])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('course_enrollments.csv', index=False)
print("✅ SUCCESS: 'course_enrollments.csv' generated.")


# ==========================================
# PART 2: SOLUTION
# ==========================================

# 1. Load Data
df = pd.read_csv('course_enrollments.csv')
courses = df.drop('Student_ID', axis=1)

# 2. Apriori
freq_courses = apriori(courses, min_support=0.2, use_colnames=True)

# 3. Rules
rules = association_rules(freq_courses, metric="lift", min_threshold=1.1)
rules['combo'] = rules['antecedents'].apply(list).astype(str) + " -> " + rules['consequents'].apply(list).astype(str)

print("\n--- Course Recommendations (Based on Lift) ---")
print(rules[['combo', 'lift', 'confidence']].sort_values('lift', ascending=False).head())

# 4. GRAPH 1: Course Popularity
plt.figure(figsize=(6,4))
courses.sum().sort_values(ascending=False).plot(kind='bar', color='royalblue')
plt.title("Course Enrollment Popularity")
plt.ylabel("Students Enrolled")
plt.tight_layout()
plt.show()

# 5. GRAPH 2: Top Recommendation Rules
plt.figure(figsize=(6,4))
top_rules = rules.sort_values('confidence', ascending=False).head(5)
plt.barh(top_rules['combo'], top_rules['confidence'], color='salmon')
plt.xlabel("Confidence (Likelihood)")
plt.title("Top Course Recommendations")
plt.tight_layout()
plt.show()

# 6. GRAPH 3: Matrix of Lift
pivot = rules.pivot(index='antecedents', columns='consequents', values='lift')
# Simplify for visualization by taking just top 10 pivot entries if large, but here small
plt.figure(figsize=(6,4))
plt.scatter(rules['support'], rules['lift'], c='green', alpha=0.7)
plt.xlabel("Support")
plt.ylabel("Lift")
plt.title("Rule Lift vs Support")
plt.grid(True)
plt.tight_layout()
plt.show()