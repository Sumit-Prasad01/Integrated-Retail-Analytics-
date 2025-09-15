import numpy as np
import pandas as pd

# Load Dataset
print("Loading Dataset ..........")
df = pd.read_csv("../../data/raw/sales data-set.csv")
print("Dataset Loaded Successfully.........")

# 1. Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

# 2. Ensure correct data types
df['Store'] = df['Store'].astype(int)
df['Dept'] = df['Dept'].astype(int)
df['Weekly_Sales'] = df['Weekly_Sales'].astype(float)

#3. Handle missing values
# Fill missing Weekly_Sales with 0 (no sales )
df['Weekly_Sales'] = df['Weekly_Sales'].fillna(0)

# Drop rows where Store or Dept is missing (critical keys)
df.dropna(subset=["Store", "Dept"], inplace=True)

# 4. Convert IsHoliday to boolean
df["IsHoliday"] = df["IsHoliday"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

# 5. Drop duplicates
df.drop_duplicates(inplace=True)

# 6. Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week
df["DayOfWeek"] = df["Date"].dt.dayofweek  # Monday=0, Sunday=6

# Lag features (per Store-Dept group)
df = df.sort_values(by=["Store", "Dept", "Date"])
df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df["lag_7"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(7)
df["lag_30"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(30)

# Rolling mean features
df["rolling_7"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["rolling_30"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.rolling(30, min_periods=1).mean())

print(" Sales dataset cleaning & preprocessing complete!")
print(df.head())

# Save cleaned dataset
df.to_csv("../../data/staging/cleaned_sales.csv", index=False)