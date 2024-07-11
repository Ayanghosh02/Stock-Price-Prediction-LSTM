# Stock-Price-Prediction-LSTM
Description:

This repository contains a stock price prediction project using LSTM (Long Short-Term Memory) neural networks. The project uses historical stock price data from Yahoo Finance, preprocesses the data for training an LSTM model, and predicts stock prices for a given company. It includes visualization of historical prices, LSTM model training, evaluation, and visualization of predictions.
Files:

    Untitled8.ipynb: Jupyter Notebook file containing the entire project code. It includes:
        Data collection using Yahoo Finance API (yfinance).
        Data preprocessing and exploration using pandas and numpy.
        Visualization of historical stock prices and trading volumes using matplotlib and seaborn.
        LSTM model creation using Keras with TensorFlow backend for predicting stock prices.
        Evaluation of model performance using RMSE (Root Mean Squared Error).
        Visualization of model predictions against actual stock prices.

Functionality:

    Data Collection: Fetches historical stock price data from Yahoo Finance for a specified company.

    Data Preprocessing: Scales the data using MinMaxScaler and prepares it for LSTM model training.

    LSTM Model Training: Constructs an LSTM model architecture using Keras Sequential API and trains it on historical stock price data.

    Prediction and Evaluation: Predicts stock prices using the trained LSTM model and evaluates prediction accuracy using RMSE.

    Visualization: Displays historical stock prices, trading volumes, model predictions, and actual prices using matplotlib and seaborn.

Usage:

To use the project:

    Open and run the Untitled8.ipynb notebook in a Jupyter environment or Google Colab.
    Ensure necessary libraries (yfinance, pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, keras) are installed.
    Follow the notebook's step-by-step instructions to collect data, preprocess it, train the LSTM model, and evaluate its performance.
    Visualize historical prices and volumes, as well as predicted vs. actual prices to understand the model's effectiveness in predicting stock prices.

Notes:

    The project uses LSTM neural networks, which are suitable for sequence prediction tasks like time series forecasting.
    Customize the project for different stocks by modifying the tech_list and adjusting parameters for the LSTM model.
    Ensure to have access to Yahoo Finance API for data collection and adhere to its usage terms.
