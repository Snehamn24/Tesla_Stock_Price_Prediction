# Tesla_Stock_Price_Prediction using Deep Learning
Tesla Stock Price prediction using the Simple RNN and LSTM and comparing the best model between Simple RNN and LSTM

## Project Overview

This project predicts Tesla stock closing prices using Deep Learning models. Historical Tesla stock data was analyzed and used to train two sequential neural network architectures:

* Simple Recurrent Neural Network (SimpleRNN)
* Long Short-Term Memory (LSTM)

The objective is to forecast future stock prices based on historical market behavior and compare the performance of both models.

---

## Problem Statement

Stock market prediction is a challenging time-series forecasting problem due to market volatility and complex dependencies.

The goal of this project is to:

* Analyze Tesla stock data
* Perform data preprocessing and feature engineering
* Build SimpleRNN and LSTM models
* Predict stock closing prices
* Compare model performance
* Deploy the best-performing model using Streamlit

---

## Dataset Information

Dataset: Tesla Historical Stock Price Dataset (TSLA.csv)

Features:

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume

Target Variable:

* Close Price

Total Records:

* 2416

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* TensorFlow / Keras
* Joblib
* Streamlit

---

## Project Workflow

### 1. Data Collection

Tesla historical stock market data was loaded using Pandas.

### 2. Exploratory Data Analysis

Performed:

* Trend Analysis
* Distribution Analysis
* Correlation Analysis
* Outlier Detection

### 3. Data Preprocessing

* Missing Value Check
* Feature Selection
* MinMax Scaling
* Sequence Generation

Lookback Window:

60 Days

Input Shape:

(60, 5)

### 4. Model Development

#### SimpleRNN

* SimpleRNN Layer (50 Units)
* Dropout Layer (0.2)
* Dense Output Layer

#### LSTM

* LSTM Layer (50 Units)
* Dropout Layer (0.2)
* Dense Output Layer

### 5. Hyperparameter Tuning

Parameters Tuned:

* Number of Units
* Dropout Rate
* Learning Rate

Best LSTM Parameters:

* Units = 64
* Dropout = 0.1
* Learning Rate = 0.001

### 6. Forecast Horizons

Predictions performed for:

* 1 Day Ahead
* 5 Days Ahead
* 10 Days Ahead

---

## Results

### Model Comparison

| Model     | RMSE  | MAE   | R² Score |
| --------- | ----- | ----- | -------- |
| SimpleRNN | 15.30 | 9.63  | 0.9598   |
| LSTM      | 34.88 | 21.21 | 0.7912   |

### Conclusion

SimpleRNN outperformed LSTM on the Tesla dataset and was selected as the final deployment model.

---

## Streamlit Application

Features:

* Upload Tesla CSV Dataset
* Display Dataset Summary
* Predict Next-Day Closing Price
* Visualize Stock Trends
* Download Prediction Report

---

## Future Enhancements

* News Sentiment Analysis
* Social Media Sentiment Integration
* GRU Implementation
* Transformer Models
* Real-Time Stock Data Integration
* Multi-Stock Forecasting

---

## Author

Sneha M N

MCA – BMS College of Engineering

Deep Learning Project – Tesla Stock Price Prediction
