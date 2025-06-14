import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import uuid

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set page config
st.set_page_config(page_title="Advanced Stock Market Predictor", layout="wide")

# Fallback LSTM model architecture
@st.cache_resource
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load or fallback to LSTM model
@st.cache_resource
def load_prediction_model():
    try:
        return load_model('Stock Predictions Model.keras')
    except Exception as e:
        st.warning("Falling back to untrained LSTM model for structure preview.")
        return build_lstm_model()

model = load_prediction_model()

# Fetch stock data
@st.cache_data(ttl=300)
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            st.error(f"No data found for {symbol}. Ensure the symbol is correct.")
            return None

        # MultiIndex fix
        if isinstance(data.columns, pd.MultiIndex):
            close_col = [col for col in data.columns if col[0].lower() == 'close']
            if not close_col:
                st.error(f"Could not find a 'Close' column. Available: {data.columns.tolist()}")
                return None
            data = data[[close_col[0]]]
            data.columns = ['Close']
        else:
            if 'close' not in [col.lower() for col in data.columns]:
                st.error(f"'Close' column missing. Available: {list(data.columns)}")
                return None
            for col in data.columns:
                if col.lower() == 'close':
                    data.rename(columns={col: 'Close'}, inplace=True)
                    break

        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Calculate moving average
def calculate_technical_indicators(df):
    if df is None or 'Close' not in df.columns:
        return None
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# Prepare sequences for prediction
def prepare_prediction_data(data_train, data_test, scaler):
    if len(data_test) < 100:
        st.error("Test data must be at least 100 rows.")
        return None, None, None
    last_100 = data_train.tail(100)
    combined = pd.concat([last_100, data_test], ignore_index=True)
    scaled = scaler.fit_transform(combined.values.reshape(-1, 1))
    x, y = [], []
    for i in range(100, len(scaled)):
        x.append(scaled[i-100:i])
        y.append(scaled[i, 0])
    return np.array(x), np.array(y), data_test.tail(len(y)).index

# Plot chart
def create_interactive_plot(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='green')))
    if 'MA50' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='red')))
    fig.update_layout(title=title, height=600, template='plotly_dark',
                      xaxis=dict(rangeslider=dict(visible=True), type="date"),
                      yaxis_title='Price ($)', hovermode='x unified')
    return fig

# UI
st.title('📈 Advanced Stock Market Predictor')

with st.sidebar:
    st.header('Settings')
    symbol = st.text_input('Stock Symbol', 'BTC-USD').strip().upper()
    if not symbol:
        st.stop()
    period = st.selectbox('Time Period', ['1y', '2y', '5y', '10y', 'max'], index=2)
    refresh_rate = st.slider('Auto-refresh (minutes)', 1, 60, 5)

end_date = datetime.now().date()
start_date = '2000-01-01' if period == 'max' else (end_date - timedelta(days=int(period[:-1]) * 365))

# Auto-refresh
st_autorefresh(interval=refresh_rate * 60 * 1000, key="datarefresh")

# Data pipeline
data = fetch_stock_data(symbol, start_date, end_date)
if data is None:
    st.stop()

data = calculate_technical_indicators(data)
if data is None:
    st.stop()

with st.expander("📄 View Raw Data"):
    st.dataframe(data.tail())

st.plotly_chart(create_interactive_plot(data, f"{symbol} Price & MA50"), use_container_width=True)

# Prepare prediction
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])
scaler = MinMaxScaler(feature_range=(0, 1))
x, y, date_index = prepare_prediction_data(data_train, data_test, scaler)

# Predict and plot
if x is not None:
    try:
        predict = model.predict(x, verbose=0)
        predict = scaler.inverse_transform(predict)
        y = scaler.inverse_transform(y.reshape(-1, 1))

        errors = np.abs(predict.flatten() - y.flatten())
        mean_error = np.mean(errors)
        ci = 1.96 * np.std(errors) / np.sqrt(len(errors))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date_index, y=y.flatten(), name='Actual', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=date_index, y=predict.flatten(), name='Predicted', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=date_index, y=predict.flatten() + ci, showlegend=False,
                                 line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=date_index, y=predict.flatten() - ci, showlegend=False,
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.2)', line=dict(color='gray', dash='dash')))

        fig.update_layout(title='Actual vs Predicted Prices', template='plotly_dark', height=500,
                          yaxis_title='Price ($)', xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error", f"${mean_error:.2f}")
        col2.metric("Confidence Interval (±)", f"${ci:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# Footer
st.markdown("""---
© 2024 Advanced Stock Market Predictor. All rights reserved.
Made with ❤️ by [A. SAI SANKEERTH](https://yourwebsite.com)
This application is for educational purposes only.
""")
