import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load dataset
try:
    df = pd.read_csv('Real estate.csv')
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("Error: File 'Real estate.csv' tidak ditemukan. Pastikan namanya benar dan ada di folder yang sama.")
    exit()

#Tentukan Variabel Independen (X) dan Dependen (y)
# X3 distance to the nearest MRT station
X = df[['X3 distance to the nearest MRT station']]
# Y house price of unit area
y = df['Y house price of unit area']

#Membagi dataset (80% training dan 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Bangun model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#Menetak nilai intercept dan koefisien regresi (slope)
intercept = model.intercept_
slope = model.coef_[0]

print("\n=== FUNGSI REGRESI ===")
print(f"Intercept (\u03b20): {intercept:.4f}")
print(f"Slope (\u03b21): {slope:.4f}")
print(f"Persamaan: Y = {intercept:.4f} + ({slope:.4f} * X)")

#Melakukan prediksi dan evaluasi performa model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== PERFORMA MODEL ===")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared: {r2:.4f}\n")

#Visualisasi Hasil Regresi
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Data Asli')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Garis Regresi')
plt.xlabel("Jarak ke Stasiun MRT (meter)")
plt.ylabel("Harga Rumah per Unit Area")
plt.title("Linear Regression: Jarak MRT vs Harga Rumah")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

#Memunculkan jendela baru berisi grafik
plt.show()