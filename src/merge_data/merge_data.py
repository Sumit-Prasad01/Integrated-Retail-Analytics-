import pandas as pd

# Load Cleaned Dataset
sales = pd.read_csv("../../data/staging/cleaned_sales.csv")
features = pd.read_csv("../../data/staging/Features_Cleaned_Data.csv")
stores = pd.read_csv("../../data/staging/cleaned_stores.csv")

# Ensure Date is datetime in all dataset
sales["Date"] = pd.to_datetime(sales['Date'])
features['Date'] = pd.to_datetime(features['Date'])

# Merge 1: Sales + Features

merged = pd.merge(
    sales,
    features,
    on = ['Store', 'Date', 'IsHoliday'],
    how = 'left'
)

final_df = pd.merge(
    merged,
    stores,
    on = 'Store',
    how = 'left'
)

# Check Final Dataset
print("Final dataset shape : ", final_df.shape)
print(final_df.head())

# Save merged dataset
final_df.to_csv("../../data/merged/merged_dataset.csv", index=False)