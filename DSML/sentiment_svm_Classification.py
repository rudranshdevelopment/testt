import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# We'll use simple numerical text features (word frequencies / sentiment score proxy)
# ==========================================
n = 220
data = {
    'review_id': range(1, n+1),
    'word_freq_good': np.random.randint(0, 8, n),
    'word_freq_bad': np.random.randint(0, 6, n),
    'exclamation_count': np.random.randint(0, 4, n),
    'review_length': np.random.randint(5, 300, n),
}
# sentiment_score proxies positive vs negative
sentiment_score = 0.4*data['word_freq_good'] - 0.6*data['word_freq_bad'] + 0.1*data['exclamation_count']
labels = np.where(np.array(sentiment_score) + np.random.normal(0,1,n) > 0, 'Positive', 'Negative')
data['sentiment'] = labels
df_gen = pd.DataFrame(data)
df_gen.to_csv('customer_reviews.csv', index=False)
print("✅ 'customer_reviews.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('customer_reviews.csv')

# 2. Features & target
X = df[['word_freq_good','word_freq_bad','exclamation_count','review_length']]
y = df['sentiment']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# 5. Predict
y_pred = svm.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Customer Sentiment (SVM) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Plot positive vs negative counts
counts = df['sentiment'].value_counts()
plt.figure(figsize=(5,4))
plt.bar(counts.index, counts.values)
plt.title("Sentiment Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
'''
