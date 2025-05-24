# evaluate_models.py

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv('House_Rent_Dataset_cleaned.csv')

# Özellik ve hedef değişken
X = df.drop('Rent', axis=1)
y = np.log1p(df['Rent'])  # Aynı log dönüşümü

# Eğitim - test bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitilmiş pipeline'ları yükle
lr_pipeline = joblib.load('linear_regression_pipeline.pkl')
xgb_pipeline = joblib.load('xgboost_pipeline.pkl')

# Tahmin yap
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)

# Log ters dönüşüm
y_test_real = np.expm1(y_test)
y_pred_lr_real = np.expm1(y_pred_lr)
y_pred_xgb_real = np.expm1(y_pred_xgb)

# Metrik fonksiyonu
def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R²: {r2:.3f}")

# Metrikleri yazdır
print_metrics("Linear Regression", y_test_real, y_pred_lr_real)
print_metrics("XGBoost", y_test_real, y_pred_xgb_real)

# Görselleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_real, y_pred_lr_real, alpha=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
plt.title("Linear Regression")
plt.xlabel("Gerçek Kira")
plt.ylabel("Tahmin")

plt.subplot(1, 2, 2)
plt.scatter(y_test_real, y_pred_xgb_real, alpha=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
plt.title("XGBoost")
plt.xlabel("Gerçek Kira")
plt.ylabel("Tahmin")

plt.tight_layout()
plt.show()
