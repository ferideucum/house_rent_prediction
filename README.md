# House Rent Prediction

Bu proje, gerçek dünya konut verileri kullanılarak kira fiyatlarını tahminleyen bir makine öğrenmesi modelinin geliştirilmesini amaçlamaktadır.

## 📊 Veri Seti

Kullanılan veri seti: **[House Rent Dataset](https://www.kaggle.com/datasets/ankurzing/salary-data)**  
Veri seti Hindistan'daki büyük şehirlerdeki konutların kira bilgilerini içermektedir.

## 🔧 Kullanılan Özellikler

- **Numerical Features**: BHK, Size, Bathroom, Size_per_BHK, Bathroom_per_BHK, Total_rooms  
- **Categorical Features**: Area Type, City, Furnishing Status  

Hedef değişken: `Rent` (log dönüşümü uygulanmıştır)

## 🧹 Ön İşleme

- Sayısal özellikler: `StandardScaler`
- Kategorik özellikler: `OneHotEncoder`
- Kira sütununa log1p dönüşümü uygulanarak dağılım normalize edilmiştir.

## 🧠 Modeller

Aşağıdaki iki regresyon modeli karşılaştırılmıştır:

1. **Linear Regression**
2. **XGBoost Regressor**

## 📈 Performans Sonuçları

| Model              | RMSE     | MAPE     | R²     |
|--------------------|----------|----------|--------|
| Linear Regression  | 8117.04  | 29.21%   | 0.678  |
| XGBoost            | 8897.72  | 32.89%   | 0.613  |

> Linear Regression modeli, daha düşük hata oranları ve daha yüksek R² değeri ile en iyi performansı göstermiştir.

## 🗃️ Kaydedilen Modeller

Eğitilen modeller `joblib` ile `.pkl` dosyaları olarak kaydedilmiştir:
models/
├── linear_regression_pipeline.pkl
└── xgboost_pipeline.pkl


## 📊 Görselleştirme

Veri keşfi ve model değerlendirmesi sırasında aşağıdaki grafikler kullanılmıştır:

- Kira dağılımı (histogram)
- Büyüklük vs Kira (scatter plot)
- Korelasyon matrisi
- Kategorik değişkenlere göre boxplot
- Gerçek vs Tahmin (iki model için scatter plot)

## 🛠️ Kullanılan Kütüphaneler

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`




