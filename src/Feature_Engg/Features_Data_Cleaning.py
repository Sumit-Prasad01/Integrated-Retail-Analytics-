import pandas as pd
import numpy as np

# Load Dataset 
print("Loading Dataset......")
df = pd.read_csv("../../data/raw/Features data set.csv")
print("Dataset Loaded Successfully......")

# 1. Convert Date to DateTime
df["Date"] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

# 2.Replace 'NA' with np.nan and convert numerics
df.replace('NA', np.nan, inplace = True)

numeric_cols = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", 
                "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]


df[numeric_cols] = df[numeric_cols].astype(float)

# 3. Handle missing values 
## Markdown columns -> 0 (no markdown applied)

markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[markdown_cols] = df[markdown_cols].fillna(0)

# Temperature, Fuel_Price, CPI, Unemployment -> fill with median
for col in ["Temperature", 'Fuel_Price', 'CPI', 'Unemployment']:
    df[col] = df[col].fillna(df[col].median())

# 4. Convert IsHoliday to boolean 
df['IsHoliday'] = df['IsHoliday'].astype(str).str.upper().map({'TRUE' : True, "FALSE" :False})

# 5. Drop duplicates (if any)
df.drop_duplicates(inplace = True)

# 6. Feature Engineering 
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday = 0 ---> Sunday = 6

# Rolling averages for key features 
for col in ["Temperature", 'Fuel_Price', 'CPI', 'Unemployment']:
    df[f'{col}_7d_avg'] = df[col].rolling(window = 7, min_periods = 1).mean()
    df[f'{col}_30d_avg'] = df[col].rolling(window = 30, min_periods = 1).mean()

print("Data Cleaning and Preprocessing Completed......")

print("Saving Cleaned Dataset")

# Save Cleaned Dataset 
df.to_csv("../../data/staging/Features_Cleaned_Data.csv")