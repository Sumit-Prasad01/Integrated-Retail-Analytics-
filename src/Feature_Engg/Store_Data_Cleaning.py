import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("../../data/raw/stores data-set.csv")

# 1. Ensure correct data types
df["Store"] = df['Store'].astype(int)
df["Type"] = df['Type'].astype(str)
df['Size'] = df['Size'].astype(float)

# 2.Handle missing values
# Fill missing type with mode (most frequent value)
if df['Type'].isnull().sum() > 0:
    df['Type'].fillna(df['Type'].mode()[0], inplace = True)

# Fillna missing size with median
if df['Size'].isnull().sum() > 0:
    df['Size'].fillna(df['Size'].median(), inplace = True)


# 3. Drop Duplicates
df.drop_duplicates(inplace = True)

# 4. Feature Engineering
# One-hot encode store type
df = pd.get_dummies(df, columns = ['Type'], prefix = "Type")

# 5. Log Transformation to reduce skewness
df['Size_log'] = np.log1p(df['Size'])

print("tores dataset cleaning & preprocessing complete!")
print(df.head())

df.to_csv("../../data/staging/cleaned_stores.csv", index=False)