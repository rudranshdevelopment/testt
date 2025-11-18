import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# ==========================================
data = {
    'image_id': range(1, 101),
    'pixel_intensity_mean': np.random.randint(50, 255, 100),
    'texture_smoothness': np.round(np.random.uniform(0.1, 0.9, 100), 3),
    'symmetry_score': np.round(np.random.uniform(0.5, 1.0, 100), 3),
    'diagnosis': np.random.choice(['Benign', 'Malignant'], 100, p=[0.7, 0.3])
}
df_gen = pd.DataFrame(data)
df_gen.to_csv('skin_cancer_data.csv', index=False)
print("✅ 'skin_cancer_data.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# ==========================================
'''
# 1. Load Dataset
df = pd.read_csv('skin_cancer_data.csv')

# 2. Features & Target
X = df[['pixel_intensity_mean', 'texture_smoothness', 'symmetry_score']]
y = df['diagnosis']

# 3. Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train SVM Model
svm_model = SVC(kernel='linear', probability=False, random_state=42)
svm_model.fit(X_train, y_train)

# 5. Predict
y_pred = svm_model.predict(X_test)

# 6. Evaluation
acc = accuracy_score(y_test, y_pred)
print("\\n--- Skin Cancer Diagnosis Results ---")
print(f"Accuracy: {acc:.2f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 7. Two-feature scatter
plt.figure(figsize=(6,4))
colors = y.map({'Benign':'green','Malignant':'red'})
plt.scatter(df['pixel_intensity_mean'], df['texture_smoothness'], c=colors, alpha=0.7)
plt.xlabel("Pixel Intensity Mean")
plt.ylabel("Texture Smoothness")
plt.title("Pixel Intensity vs Texture (colored by diagnosis)")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
