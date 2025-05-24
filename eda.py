
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
df = pd.read_csv('House_Rent_Dataset_cleaned.csv')

# 1. Kira Dağılımı (Histogram)
#Kira değerlerinin nasıl dağıldığını gör:
plt.figure(figsize=(8,5))
sns.histplot(df['Rent'], bins=50, kde=True)
plt.title("Kira Dağılımı")
plt.xlabel("Kira")
plt.ylabel("Frekans")
plt.show()

#📈 2. Kira vs Özellik (Scatter Plot)
#Örneğin Size ile Rent arasındaki ilişkiyi incele:
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='Size', y='Rent', hue='City', alpha=0.6)
plt.title("Ev Büyüklüğü vs Kira")
plt.show()


#🔲 3. Korelasyon Matrisi (Isı Haritası)
#Sayısal değişkenler arasındaki ilişkileri analiz etmek için:

plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()


#🧱 4. Kategorik Özelliklere Göre Ortalama Kira (Boxplot)
#Örneğin Furnishing Status'a göre kira:
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Furnishing Status', y='Rent')
plt.title("Eşyalı Durumuna Göre Kira Dağılımı")
plt.show()