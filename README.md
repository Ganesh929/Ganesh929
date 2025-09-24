# 📈 Stock Price Prediction App

This is a **Streamlit web app** that predicts stock prices using an LSTM deep learning model.  
It fetches real-time stock data from **Yahoo Finance** and shows predicted prices alongside actual prices.

---

## 🚀 Features
- Enter any stock ticker (e.g., `AAPL`, `TSLA`, `MSFT`)
- Select time period (`1y`, `2y`, `5y`, `10y`)
- Download & display stock data
- Predict stock trends using trained model
- Interactive chart (Actual vs Predicted)

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) - Web app
- [TensorFlow/Keras](https://www.tensorflow.org/) - LSTM model
- [Scikit-learn](https://scikit-learn.org/) - Preprocessing
- [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/) - Stock data
- Pandas, NumPy, Matplotlib

---

## 📂 Project Structure
Ganesh929/                  <- Your GitHub repo (name can be anything)
│
├── app.py                  <- Main Streamlit app file (must match in deploy settings)
├── requirements.txt        <- List of Python dependencies
├── README.md               <- Project description
│
├── stock_model.h5          <- (Optional) Trained LSTM model file if you are not training inside app.py
├── scaler.pkl              <- (Optional) Scaler used for preprocessing
