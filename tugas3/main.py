import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'new-york_9-24-2016_9-30-2017.csv')

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File tidak ditemukan. Pastikan nama dan lokasi file benar.")
    exit()

# Mengubah 'Date' menjadi urutan hari (1-365) agar bisa diolah secara matematis
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear

# Membuat kolom target 'Average Price' dari High Price dan Low Price
df['Average Price'] = (df['High Price'] + df['Low Price']) / 2

# Mengurutkan data berdasarkan hari agar plot garisnya nanti rapi
df = df.sort_values(by='DayOfYear')

X = df[['DayOfYear']]
y = df['Average Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("=== PERFORMA MODEL POLYNOMIAL ===")
print(f"RMSE (Testing): {rmse_test:.4f}")
print(f"R-squared: {r2_test:.4f}")

plt.scatter(X, y, color='blue', alpha=0.5, label='Data Harga Asli')

# Membuat garis lengkung prediksi yang mulus
X_plot = pd.DataFrame({'DayOfYear': np.linspace(X['DayOfYear'].min(), X['DayOfYear'].max(), 100)})
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Regression (Degree 3)')
plt.xlabel("Hari Ke- (Day of Year)")
plt.ylabel("Harga Rata-rata Labu (USD)")
plt.title("Prediksi Harga Labu di New York (Polynomial Regression)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()