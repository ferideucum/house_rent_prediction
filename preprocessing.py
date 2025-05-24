# Veri temizleme ve ön işleme


import numpy as np

import pandas as pd


df = pd.read_csv('House_Rent_Dataset.csv')

# Gereksiz sütunları sil
df.drop(["Posted On", "Tenant Preferred", "Point of Contact", "Floor", "Area Locality"], axis=1, inplace=True)

# Eksik verileri doldur
df.fillna(0, inplace=True)

# Özellik mühendisliği
df['Size_per_BHK'] = df['Size'] / df['BHK']
df['Bathroom_per_BHK'] = df['Bathroom'] / df['BHK']
df['Total_rooms'] = df['BHK'] + df['Bathroom']


# Kategorik verileri dönüştür
df['Area Type'] = df['Area Type'].astype('category')
df['City'] = df['City'].astype('category')
df['Furnishing Status'] = df['Furnishing Status'].astype('category')

# Sayısal verileri tip dönüşümü
df['BHK'] = df['BHK'].astype('int')
df['Rent'] = df['Rent'].astype('int')
df['Size'] = df['Size'].astype('int')
df['Bathroom'] = df['Bathroom'].astype('int')

# Kategorik ve sayısal özellikler
categorical_features = ['Area Type', 'City', 'Furnishing Status']
numerical_features = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK', 'Bathroom_per_BHK', 'Total_rooms']



# IQR yöntemiyle aykırı değerleri filtrele
columns = df.select_dtypes(include=[np.number]).columns
min_values = []
max_values = []
for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    min_values.append(min_value)
    max_values.append(max_value)
    print(f"Column: {column}, min: {min_value}, max: {max_value}")

# Aykırı değerleri sil
for i, column in enumerate(columns):
    df = df[(df[column] >= min_values[i]) & (df[column] <= max_values[i])]

# Temiz veri özetini yazdır
print(df.describe())


df.to_csv(('House_Rent_Dataset_cleaned.csv'),index=False)

