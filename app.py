import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load your data
data = pd.read_csv("btc_data.csv", index_col="Date", parse_dates=True)

# Fit the ARIMA model (adjust (p, d, q) accordingly)
model_fit = joblib.load("model_arima.pkl")

def make_forecast(model_fit, forecast_period=1):
    forecast = model_fit.forecast(steps=forecast_period)
    last_date = data.index[-1]  # Use the last date from your data
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecasted Price'])
    return forecast_df

# Streamlit App
st.title("Bitcoin Price Prediction")
st.write("This app uses an ARIMA model to predict the next day's Bitcoin price.")

# Display the actual data
st.line_chart(data)

# Make the forecast
forecast_period = 1
forecast_df = make_forecast(model_fit, forecast_period)

# Display forecast results
st.write("Forecasted Price for the next day:")
st.write(forecast_df)

# Optionally plot the forecast alongside historical data
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Actual Price')
plt.plot(forecast_df.index, forecast_df['Forecasted Price'], marker='o', label='Forecasted Price', color='orange')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
st.pyplot(plt)
