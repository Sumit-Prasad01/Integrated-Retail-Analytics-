import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint, uniform


# Load dataset

df = pd.read_csv("../data/final_data/merged_top10_dataset.csv")

# Target variable
y = df["Weekly_Sales"]

# Drop target from features
X = df.drop(columns=["Weekly_Sales"], errors="ignore")

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Preprocessing pipeline

preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),        # Handle NaNs
    ("power", PowerTransformer(method="yeo-johnson")),   # Fix skewness
    ("scaler", StandardScaler())                         # Scale features
])


# Gradient Boosting Model + Randomized Search

model = GradientBoostingRegressor(random_state=42)

param_distributions = {
    "model__n_estimators": randint(100, 500),
    "model__learning_rate": uniform(0.01, 0.3),
    "model__max_depth": randint(3, 10),
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 10)
}

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=30,                        # number of random configs to try
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42
)


# Train the model

print("Training with RandomizedSearchCV...")
random_search.fit(X_train, y_train)


# Best Model

print("\n Best Hyperparameters:", random_search.best_params_)

best_model = random_search.best_estimator_


# Evaluate on Test Set

y_pred = best_model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Final Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")


# Save Best Model

joblib.dump(best_model, "../models/best_model.pkl")
print("\n Best model saved at models/best_model.pkl")
