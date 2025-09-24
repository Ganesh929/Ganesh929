import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load model & scaler
model = load_model("stock_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("📈 Stock Price Prediction App")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT)", "AAPL")
period = st.selectbox("Select period", ["1y", "2y", "5y", "10y"])

if st.button("Predict"):
    # Fetch data
    data = yf.download(ticker, period=period, interval="1d")
    st.subheader("Recent Stock Data")
    st.write(data.tail())

    # Preprocess
    prices = data['Close'].values.reshape(-1,1)
    scaled_data = scaler.transform(prices)

    X = []
    n_steps = 60
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
    X = np.array(X).reshape(-1, n_steps, 1)

    # Predict
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Plot
    st.subheader("Predicted vs Actual Prices")
    fig, ax = plt.subplots()
    ax.plot(data.index[n_steps:], prices[n_steps:], label="Actual")
    ax.plot(data.index[n_steps:], predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)
