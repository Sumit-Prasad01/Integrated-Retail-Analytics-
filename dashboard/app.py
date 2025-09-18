import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load Data & Model
# -----------------------
DATA_PATH = "../data/final_data/merged_top10_dataset.csv"
MODEL_PATH = "../models/best_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🛒 Retail Analytics")
menu = ["Home", "EDA", "Prediction"]
choice = st.sidebar.radio("Navigation", menu)

# -----------------------
# Home Page
# -----------------------
if choice == "Home":
    st.title("🛒 Retail Analytics & Sales Forecasting")
    st.write("""
    This app predicts **Weekly Sales** for retail stores using a trained 
    **Gradient Boosting model** on ~421k rows of historical data.

    ### Features:
    - 📊 Explore dataset & simple trends  
    - 🤖 Predict weekly sales with ML model  
    """)

# -----------------------
# EDA Page
# -----------------------
elif choice == "EDA":
    st.title("📊 Data Exploration")
    st.write("Preview of dataset:")
    st.dataframe(df.head(20))

    st.subheader("Dataset Columns")
    st.write(list(df.columns))

    # Average Weekly Sales by Department
    if "Dept" in df.columns:
        st.subheader("Average Weekly Sales by Department")
        st.bar_chart(df.groupby("Dept")["Weekly_Sales"].mean())

    # Weekly trend using Week_x
    if "Week_x" in df.columns:
        st.subheader("Average Weekly Sales by Week_x")
        st.line_chart(df.groupby("Week_x")["Weekly_Sales"].mean())

    # Distribution of Weekly Sales
    st.subheader("Distribution of Weekly Sales")
    st.bar_chart(df["Weekly_Sales"].value_counts().sort_index())

# -----------------------
# Prediction Page
# -----------------------
elif choice == "Prediction":
    st.title("🤖 Weekly Sales Prediction")

    st.write("Enter feature values to predict weekly sales:")

    # Input fields for the 10 features
    rolling_7 = st.number_input("Rolling 7-day Sales", value=20000.0)
    lag_1 = st.number_input("Lag 1-day Sales", value=21000.0)
    lag_7 = st.number_input("Lag 7-day Sales", value=20500.0)
    rolling_30 = st.number_input("Rolling 30-day Sales", value=22000.0)
    markdown3 = st.number_input("MarkDown3", value=0.0)
    week_x = st.number_input("Week_x", min_value=1, max_value=52, value=10)
    week_y = st.number_input("Week_y", min_value=1, max_value=52, value=10)
    lag_30 = st.number_input("Lag 30-day Sales", value=21500.0)
    temp_30d = st.number_input("Temperature 30d Avg", value=65.0)
    dept = st.number_input("Department ID", min_value=1, value=1)

    if st.button("Predict"):
        input_df = pd.DataFrame([[rolling_7, lag_1, lag_7, rolling_30, markdown3,
                                  week_x, week_y, lag_30, temp_30d, dept]],
                                columns=["rolling_7", "lag_1", "lag_7", "rolling_30",
                                         "MarkDown3", "Week_x", "Week_y", "lag_30",
                                         "Temperature_30d_avg", "Dept"])
        
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Weekly Sales: **${prediction:,.2f}**")