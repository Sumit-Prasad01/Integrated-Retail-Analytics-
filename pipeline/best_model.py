import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("../data/final_data/merged_top10_dataset.csv")

# Target
y = df["Weekly_Sales"]

# Drop unwanted/non-numeric cols
drop_cols = ["Weekly_Sales", "Unnamed: 0", "Weekly_Sales_log", "Date"]
X = df.drop(columns=drop_cols, errors="ignore")

# Keep only numeric features
X = X.select_dtypes(include=["int64", "float64"])

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Preprocessing Pipeline
# -----------------------
numeric_features = X.columns.tolist()

preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),        # Handle NaNs
    ("power", PowerTransformer(method="yeo-johnson")),   # Handle skewness
    ("scaler", StandardScaler())                         # Scale features
])

# -----------------------
# Define models
# -----------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Support Vector Regressor": SVR(kernel="rbf", C=100, epsilon=0.1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

# -----------------------
# Train & Evaluate
# -----------------------
results = []

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\n Model Performance:")
print(results_df)

# Save results
results_df.to_csv("model_results.csv", index=False)
