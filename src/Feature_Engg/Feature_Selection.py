import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../../data/merged/merged_dataset.csv")

# Target variable
y = df["Weekly_Sales"]

# Drop unwanted columns (target + junk + non-numeric like Date)
drop_cols = ["Weekly_Sales", "Unnamed: 0", "Weekly_Sales_log", "Date"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Keep only numeric columns
X = X.select_dtypes(include=["int64", "float64"])


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train RandomForest for feature importance

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feat_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

# Sort & select top 10
top10_features = feat_importances.sort_values(by="Importance", ascending=False).head(10)

print("Top 10 Features:")
print(top10_features)


# Save top 10 features list

top10_features.to_csv("../../data/final_data/top10_features.csv", index=False)


# Create new dataset with top 10 features + target

selected_features = top10_features["Feature"].tolist()
df_top10 = df[selected_features + ["Weekly_Sales"]]

# Save the reduced dataset
df_top10.to_csv("../../data/final_data/merged_top10_dataset.csv", index=False)

print("Saved top 10 features dataset at ../../data/final_data/merged_top10_dataset.csv")
