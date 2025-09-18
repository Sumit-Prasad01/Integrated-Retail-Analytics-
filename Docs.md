
# 🛠️ Developer Documentation - Retail Analytics & Sales Forecasting

This document provides **developer-level details** for maintaining and extending the Retail Analytics project.

---

## 📌 Setup Instructions

### 1. Clone Repository

- GitHub Link : https://github.com/Sumit-Prasad01/Integrated-Retail-Analytics-

```bash
git clone <your-repo-url>
cd Retail-Analytics
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
cd dashboard
streamlit run app.py
```

---

## 📊 Data Pipeline

### 1. Raw Datasets
- `sales.csv` → Weekly sales for stores/departments  
- `features.csv` → Holidays, temperature, fuel, markdowns  
- `stores.csv` → Store type and size  

### 2. Data Preprocessing
- Handle **NaNs** using `SimpleImputer`
- Fix **skewness** using `PowerTransformer`
- Scale features using `StandardScaler`

### 3. Feature Engineering
- Lags (`lag_1`, `lag_7`, `lag_30`)  
- Rolling windows (`rolling_7`, `rolling_30`)  
- Aggregates (`Temperature_30d_avg`)  
- Markdown variables (e.g., `MarkDown3`)  

### 4. Final Dataset
- Top 10 features selected using **RandomForest feature importance**  
- Saved as:  
  `data/final_data/merged_top10_dataset.csv`

---

## 🤖 Modeling Pipeline

### 1. Models Evaluated
- Linear Regression, Ridge, Lasso  
- KNN Regressor  
- Support Vector Regressor (SVR)  
- Gradient Boosting Regressor ✅ (best choice)  

### 2. Hyperparameter Tuning
- Used **RandomizedSearchCV** for Gradient Boosting  
- Parameters tuned:
  ```
  n_estimators, learning_rate, max_depth,
  min_samples_split, min_samples_leaf
  ```

### 3. Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R²** (Coefficient of Determination)  

### 4. Best Model
- Gradient Boosting Regressor with tuned hyperparameters  
- Saved as:  
  `models/best_model.pkl`

---

## 🖥️ Streamlit Dashboard

### Pages
1. **Home** → Overview of project  
2. **EDA** → Dataset preview, weekly trends, department-level analysis  
3. **Prediction** → Input 10 features → get weekly sales prediction  

### Example Input Features
```
rolling_7, lag_1, lag_7, rolling_30, MarkDown3,
Week_x, Week_y, lag_30, Temperature_30d_avg, Dept
```

### Example Prediction Output
```
Predicted Weekly Sales: $25,436.75
```

---

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


---
## 🔮 Future Enhancements
- Add **LightGBM/CatBoost** for faster training  
- Batch prediction via **CSV upload in Streamlit**  
- Deploy to **Streamlit Cloud / Heroku**  
- Add **automated retraining pipeline**  

---
💻 **Maintainer Notes**  
- Always retrain model when new sales data is available.  
- Validate features after preprocessing to ensure correct order.  
- Keep model + dataset paths consistent in `app.py`.  
---


## Why This Project Matters for Businesses

Retail companies (like Walmart, Target, or Reliance Retail) operate with tight margins and massive inventories.
Accurately forecasting weekly sales is crucial for survival and growth.

🔹 Business Benefits
## 1. 📦 Inventory Optimization

- Predicting demand ensures the right amount of stock is available.

- Avoids overstocking (ties up capital, increases storage costs) and understocking (lost sales, dissatisfied customers).

#### - Example:
- If a store knows Dept X will sell 20% more next week due to a holiday, it can increase supply accordingly.

## 2. 🎯 Data-Driven Promotions

- Insights from markdowns (MarkDown3, etc.) show how discounts impact sales.

- Helps businesses decide when & where to offer promotions.

#### Example:
- Markdown campaigns can be targeted at low-performing weeks to smoothen revenue.

## 3. 🏬 Store & Department Performance Monitoring

- Identifies which stores/departments consistently outperform others.

- Helps allocate resources, staff, and marketing budgets more effectively.

#### Example:
- Store A sells more electronics, while Store B sells more groceries → promotions can be tailored.

## 4. ⏱ Seasonal & Holiday Planning

- Features like Week_x, IsHoliday, and lag/rolling averages capture seasonality.

- Businesses can prepare in advance for high-demand periods (festivals, Black Friday, Diwali).

#### Example:
- Boost staff hiring and logistics during predictable sales spikes.

## 5. 💰 Financial Forecasting & Strategic Decisions

- Reliable sales predictions support budget planning and profit margin analysis.

- Can simulate “what-if” scenarios → e.g., What if fuel prices rise by 10%?

#### Example:
- Management can forecast impact of inflation on sales and adjust strategies.

## 🔹 Real-World Impact

- 📉 Reduce waste: perishable goods don’t sit unsold.

- 📈 Increase revenue: fewer stock-outs → more satisfied customers.

- 💵 Save costs: optimize markdowns, staffing, logistics.

- 🎯 Better decisions: management makes choices backed by data, not gut feeling.

## ✅ In short:
- This project turns raw sales data into actionable insights that help retail businesses predict demand, optimize operations, reduce costs, and maximize profits.