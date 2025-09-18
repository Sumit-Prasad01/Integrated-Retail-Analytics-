
# 🛒 Retail Analytics & Sales Forecasting

This project is an **end-to-end Machine Learning pipeline and dashboard** for retail sales forecasting using historical data from multiple stores, departments, and features.

## 📌 Project Overview
The goal of this project is to:
- Analyze and preprocess sales, store, and feature datasets
- Perform **feature engineering** (lags, rolling averages, markdowns, seasonal effects, etc.)
- Select the **top features** for prediction
- Build and tune **regression models** (Gradient Boosting, KNN, SVR, etc.)
- Evaluate models using **RMSE, MAE, and R²**
- Deploy the final model in a **Streamlit dashboard** for interactive predictions and visualizations

## 📊 Dataset
The dataset used includes:
- **Sales Data** → Weekly sales per store & department  
- **Features Data** → Holidays, temperature, fuel price, markdowns, etc.  
- **Store Data** → Store type, size, etc.  

After merging and feature selection, the final dataset includes **10 top features**:
```
rolling_7, lag_1, lag_7, rolling_30, MarkDown3, 
Week_x, Week_y, lag_30, Temperature_30d_avg, Dept
```

with the target variable:
```
Weekly_Sales
```

## ⚙️ Project Pipeline
1. **Data Cleaning & Preprocessing**
   - Handle missing values, outliers, skewness
   - Merge datasets (sales, stores, features)
2. **EDA (Exploratory Data Analysis)**
   - Visualize trends, distributions, seasonal patterns
3. **Feature Engineering**
   - Lags, rolling averages, holiday features
   - Feature importance ranking
4. **Modeling**
   - Models tested: KNN, SVR, Ridge/Lasso, Gradient Boosting
   - Hyperparameter tuning using **RandomizedSearchCV**
   - Evaluation metrics: RMSE, MAE, R²
5. **Streamlit Dashboard**
   - Home: Project overview
   - EDA: Dataset preview & simple charts
   - Prediction: Input features & get weekly sales forecast

## 🚀 Results
- **Best Models:**  
  - KNN Regressor (Best RMSE, but slower at scale)  
  - Gradient Boosting (Balanced accuracy & scalability ✅)

- Example Performance:  
  - RMSE: ~4068  
  - R²: ~0.968  

## 🖥️ Streamlit App
Run the app with:
```bash
streamlit run app.py
```

### App Features:
- **Home Page** → Overview of project  
- **EDA Page** → Explore dataset, charts  
- **Prediction Page** → Input 10 features & predict weekly sales  

## 📦 Project Structure
```
Retail-Analytics/
│── data/
│   ├── raw/                # Raw datasets (features, sales, stores)
│   ├── merged/             # Merged dataset
│   └── final_data/         # Final dataset with top 10 features
│── models/
│   └── best_model.pkl      # Saved trained model
│── dashboard/
│   └── app.py              # Streamlit app
│── pipeline/               # Model training pipeline
│── src/                    # Other python scripts
│── notebooks/              # EDA, feature engineering, model training
│── README.md               # Project documentation
```

