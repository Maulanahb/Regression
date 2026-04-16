# =========================
# 1. Import Library
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 2. Load Dataset
# =========================
# Pastikan file voice.csv ada di folder yang sama
df = pd.read_csv('voice.csv')

print("=== HEAD DATA ===")
print(df.head())

# =========================
# 3. Exploratory Data Analysis
# =========================
print("\n=== INFO DATA ===")
print(df.info())

print("\n=== DESKRIPSI DATA ===")
print(df.describe())

print("\n=== DISTRIBUSI LABEL ===")
print(df['label'].value_counts())

# =========================
# 4. Pisahkan Fitur & Label
# =========================
X = df.drop('label', axis=1)
y = df['label']

# =========================
# 5. Encoding Label
# =========================
le = LabelEncoder()
y = le.fit_transform(y)   # male=1, female=0

# =========================
# 6. Normalisasi Data
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 7. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# =========================
# 8. Training Model
# =========================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# 9. Prediksi
# =========================
y_pred = model.predict(X_test)

print("\n=== HASIL PREDIKSI (10 DATA PERTAMA) ===")
print(y_pred[:10])

# =========================
# 10. Evaluasi Model
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)