import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# Features are simple numeric proxies (word frequencies, presence flags)
# ==========================================
n = 300
data = {
    'email_id': range(1, n+1),
    'word_freq_free': np.random.randint(0, 10, n),
    'word_freq_offer': np.random.randint(0, 8, n),
    'word_freq_click': np.random.randint(0, 6, n),
    'has_attachment': np.random.choice([0,1], n, p=[0.8,0.2]),
    'email_length': np.random.randint(20, 2000, n),
}
# Build label - more likely spam if frequent spammy words are present
spam_prob = 0.3 + 0.12*data['word_freq_free'] + 0.1*data['word_freq_offer'] + 0.08*data['word_freq_click']
labels = np.where(np.random.rand(n) < (spam_prob/3.0), 'Spam', 'Ham')
data['label'] = labels
df_gen = pd.DataFrame(data)
df_gen.to_csv('email_spam.csv', index=False)
print("✅ 'email_spam.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load
df = pd.read_csv('email_spam.csv')

# 2. Features & target
X = df[['word_freq_free','word_freq_offer','word_freq_click','has_attachment','email_length']]
y = df['label']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 4. Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# 5. Predict
y_pred = nb.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print("\\n--- Email Spam Detection (Naive Bayes) ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=['Ham','Spam'])
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0,1], ['Ham','Spam'])
plt.yticks([0,1], ['Ham','Spam'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.tight_layout()
plt.show()
'''
