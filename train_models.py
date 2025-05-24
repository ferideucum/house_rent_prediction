# Modelleri eğitme ve karşılaştırma

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np



# Veri setini yükle
df = pd.read_csv('House_Rent_Dataset_cleaned.csv')


categorical_features = ['Area Type', 'City', 'Furnishing Status']
numerical_features = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK', 'Bathroom_per_BHK', 'Total_rooms']

# Özellik ve hedef değişkenler
X = df.drop('Rent', axis=1)
y = np.log1p(df['Rent'])  # log dönüşümü

# Eğitim - test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ön işleme pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Linear Regression pipeline
lr_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Modelleri eğit
lr_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

# Tahminler
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)

# Performans metrikleri (orijinal scale için ters dönüşüm)
y_test_real = np.expm1(y_test)
y_pred_lr_real = np.expm1(y_pred_lr)
y_pred_xgb_real = np.expm1(y_pred_xgb)

def print_metrics(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R²: {r2:.3f}")

print_metrics("Linear Regression", y_test_real, y_pred_lr_real)
print_metrics("XGBoost", y_test_real, y_pred_xgb_real)

import joblib

# Pipeline'ları kaydet
joblib.dump(lr_pipeline, 'linear_regression_pipeline.pkl')
joblib.dump(xgb_pipeline, 'xgboost_pipeline.pkl')

