import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

# Load Dataset
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/RossmannStoreSales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Streamlit App
st.set_page_config(page_title='Sales Forecasting', page_icon='📈', layout='wide')
st.title("📈 Sales Price Prediction & Demand Forecasting")

# Sidebar
st.sidebar.header("User Input")
selected_store = st.sidebar.selectbox("Select Store ID", df['Store'].unique())

# Filter data
df_store = df[df['Store'] == selected_store]
st.subheader(f"Store {selected_store} Sales Data")
st.dataframe(df_store.head())

# Data Visualization
st.subheader("Sales Trend Over Time")
fig = px.line(df_store, x='Date', y='Sales', title=f"Sales Trend for Store {selected_store}")
st.plotly_chart(fig)

# Train a Prediction Model
st.subheader("Sales Prediction using Machine Learning")

df_train = df_store[['Sales', 'Customers']].dropna()
X = df_train.drop('Sales', axis=1)
y = df_train['Sales']

model = RandomForestRegressor()
model.fit(X, y)
joblib.dump(model, 'sales_model.pkl')

new_customers = st.slider("Enter Expected Number of Customers", min_value=int(X.min()), max_value=int(X.max()))
if st.button("Predict Sales"):
    prediction = model.predict([[new_customers]])
    st.success(f"Predicted Sales: ${prediction[0]:,.2f}")

# Demand Forecasting using Prophet
st.subheader("Demand Forecasting with Prophet")

df_forecast = df_store[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig2 = px.line(forecast, x='ds', y='yhat', title=f"Forecasted Sales for Store {selected_store}")
st.plotly_chart(fig2)
