import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title("Tesla Stock Price Prediction")
st.write(
    "This application predicts the next-day Tesla closing price using the best-performing SimpleRNN model."
)

# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = load_model("final_tesla_model.keras",compile=False)
scaler = joblib.load("scaler.pkl")

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
LOOKBACK = 60
CLOSE_INDEX = FEATURES.index("Close")

# ==========================================
# MODEL PERFORMANCE DETAILS
# ==========================================

st.subheader("Model Used")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected Model", "SimpleRNN")
col2.metric("RMSE", "15.30")
col3.metric("MAE", "9.63")
col4.metric("R² Score", "0.9598")

st.info(
    "SimpleRNN was selected for deployment because it achieved the lowest RMSE and highest R² score compared to LSTM."
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload TSLA.csv file",
    type=["csv"]
)

# ==========================================
# HELPER FUNCTION
# ==========================================

def inverse_close_value(value):
    """
    Converts scaled predicted close price back to original price.
    """
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, CLOSE_INDEX] = value
    inverse = scaler.inverse_transform(dummy)
    return inverse[0, CLOSE_INDEX]

# ==========================================
# MAIN APP
# ==========================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort data by date
    df = df.sort_values("Date")

    # Set date as index
    df.set_index("Date", inplace=True)

    # Check required columns
    missing_columns = [col for col in FEATURES if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in uploaded file: {missing_columns}")

    elif len(df) < LOOKBACK:
        st.error(
            f"Dataset must contain at least {LOOKBACK} rows for prediction."
        )

    else:
        # ==========================================
        # DATASET SUMMARY
        # ==========================================

        st.subheader("Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Records", len(df))
        col2.metric("Start Date", str(df.index.min().date()))
        col3.metric("End Date", str(df.index.max().date()))
        col4.metric("Last Close Price", f"${df['Close'].iloc[-1]:.2f}")

        # ==========================================
        # PREPARE DATA FOR PREDICTION
        # ==========================================

        data = df[FEATURES]

        scaled_data = scaler.transform(data)

        last_60_days = scaled_data[-LOOKBACK:]

        X_input = np.array([last_60_days])

        prediction_scaled = model.predict(X_input)

        predicted_price = inverse_close_value(
            prediction_scaled[0][0]
        )

        last_actual_price = df["Close"].iloc[-1]

        price_difference = predicted_price - last_actual_price

        percentage_change = (
            price_difference / last_actual_price
        ) * 100

        # ==========================================
        # PREDICTION OUTPUT
        # ==========================================

        st.subheader("Next-Day Prediction")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Last Actual Closing Price",
            f"${last_actual_price:.2f}"
        )

        col2.metric(
            "Predicted Next-Day Closing Price",
            f"${predicted_price:.2f}"
        )

        col3.metric(
            "Predicted Change",
            f"{percentage_change:.2f}%",
            f"${price_difference:.2f}"
        )

        st.write(
            "The prediction is generated using the last 60 trading days from the uploaded dataset."
        )

        # ==========================================
        # LAST 100 DAYS + PREDICTED POINT GRAPH
        # ==========================================

        st.subheader("Last 100 Days Closing Price with Prediction")

        recent_df = df.tail(100).copy()

        next_day = recent_df.index[-1] + pd.Timedelta(days=1)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            recent_df.index,
            recent_df["Close"],
            label="Actual Close Price"
        )

        ax.scatter(
            next_day,
            predicted_price,
            label="Predicted Next-Day Price",
            s=100
        )

        ax.set_title(
            "Tesla Closing Price: Last 100 Days + Predicted Next Day"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price ($)")
        ax.legend()

        st.pyplot(fig)

        # ==========================================
        # FULL CLOSING PRICE GRAPH
        # ==========================================

        st.subheader("Full Tesla Closing Price Trend")

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(df.index, df["Close"])

        ax2.set_title("Tesla Closing Price Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Closing Price ($)")

        st.pyplot(fig2)

        # ==========================================
        # DOWNLOADABLE PREDICTION REPORT
        # ==========================================

        report_df = pd.DataFrame({
            "Model": ["SimpleRNN"],
            "Lookback Days": [LOOKBACK],
            "Last Dataset Date": [str(df.index.max().date())],
            "Last Actual Close": [last_actual_price],
            "Predicted Next-Day Close": [predicted_price],
            "Predicted Change": [price_difference],
            "Predicted Change %": [percentage_change],
            "RMSE": [15.30],
            "MAE": [9.63],
            "R2 Score": [0.9598]
        })

        csv = report_df.to_csv(index=False)

        st.download_button(
            label="Download Prediction Report",
            data=csv,
            file_name="tesla_prediction_report.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload TSLA.csv to generate prediction.")