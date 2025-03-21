import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
def load_data():
    df = pd.read_csv('retail_data.csv', sep=';')  # Update with actual dataset path
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert 'StateHoliday' to numeric, replacing non-numeric values with 0
    df['StateHoliday'] = pd.to_numeric(df['StateHoliday'], errors='coerce').fillna(0).astype(int)
    
    # Ensure 'Petrol_price' is converted to float
    df['Petrol_price'] = pd.to_numeric(df['Petrol_price'], errors='coerce').fillna(df['Petrol_price'].median())
    
    return df

df = load_data()

# Streamlit App
st.set_page_config(page_title='Sales Forecasting', page_icon='📈', layout='wide')
st.title("📈 Sales Price Prediction & Demand Forecasting")

# Dataset Overview
st.subheader("Dataset Overview")
st.write("This dataset contains historical sales and demand data, including various factors influencing demand such as promotions, holidays, and fuel prices. The dataset consists of the following columns:")
st.markdown("""
- **Product_id**: Unique identifier for the product.
- **Product_Code**: Code assigned to the product.
- **Warehouse**: The storage location of the product.
- **Product_Category**: The category to which the product belongs.
- **Date**: The date of the sales record.
- **Order_Demand**: The total demand for the product on that date.
- **Open**: Whether the store was open on that date (1 = Yes, 0 = No).
- **Promo**: Indicates if there was a promotional offer (1 = Yes, 0 = No).
- **StateHoliday**: Whether the date was a state holiday (1 = Yes, 0 = No).
- **SchoolHoliday**: Whether the date was a school holiday (1 = Yes, 0 = No).
- **Petrol_price**: The price of petrol on that date, which can influence demand.
""")

# Sample Data
st.subheader("Sample Data")
st.write("Here is an example row from the dataset:")
sample_data = {"Product_id": [786725], "Product_Code": ["Product_0033"], "Warehouse": ["Whse_S"], 
               "Product_Category": ["Category_005"], "Date": ["01/03/2016"], "Order_Demand": [16000],
               "Open": [1], "Promo": [0], "StateHoliday": [0], "SchoolHoliday": [0], "Petrol_price": [91]}
st.dataframe(pd.DataFrame(sample_data))

# Select Warehouse and Product Category
warehouses = df['Warehouse'].unique()
categories = df['Product_Category'].unique()

selected_warehouse = st.selectbox("Select Warehouse", warehouses)
selected_category = st.selectbox("Select Product Category", categories)

# Filter data
df_filtered = df[(df['Warehouse'] == selected_warehouse) & (df['Product_Category'] == selected_category)]
st.subheader(f"Sales Data for Warehouse: {selected_warehouse} | Category: {selected_category}")
st.dataframe(df_filtered.head())

# Data Visualization
st.subheader("Order Demand Trend Over Time")
fig = px.line(df_filtered, x='Date', y='Order_Demand', title=f"Order Demand Trend for {selected_category} in {selected_warehouse}")
st.plotly_chart(fig)

# Train a Prediction Model
st.subheader("Order Demand Prediction using Machine Learning")

df_train = df_filtered[['Order_Demand', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Petrol_price']].dropna()
X = df_train.drop('Order_Demand', axis=1)
y = df_train['Order_Demand']

model = RandomForestRegressor()
model.fit(X, y)

open_store = st.selectbox("Is the store open?", [0, 1])
promo = st.selectbox("Is there a promo?", [0, 1])
state_holiday = st.selectbox("State Holiday", df['StateHoliday'].unique())
school_holiday = st.selectbox("Is it a school holiday?", [0, 1])
petrol_price = st.slider("Petrol Price", min_value=float(df['Petrol_price'].min()), max_value=float(df['Petrol_price'].max()))

if st.button("Predict Order Demand"):
    prediction = model.predict([[open_store, promo, state_holiday, school_holiday, petrol_price]])
    st.success(f"Predicted Order Demand: {prediction[0]:,.2f}")
