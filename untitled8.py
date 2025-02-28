# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Jue0UqMIMiSDE6oBmmlo-O-9BZwDRG8o
"""

!pip install -q yfinance

from google.colab import files


uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['SBUX']

# Set up End and Start times for data grab
#tech_list = ['SBUX']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)


company_list = [SBUX]
company_name = ["starbucks"]

for company, com_name in zip(company_list, company_name):
    company['company_name'] = com_name

df = pd.concat(company_list, axis=0)
df.tail(10)

# Summary Stats
SBUX.describe()

# General info
SBUX.info()

# Let's see a historical view of the closing price
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}")

plt.tight_layout()

# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i - 1]}")

plt.tight_layout()

import matplotlib.pyplot as plt

# Assuming you have the historical data for one company in the variable "company"
plt.figure(figsize=(10, 6))
company['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)

plt.title("Sales Volume for SBUX")
plt.tight_layout()
plt.show()

#import matplotlib.pyplot as plt

ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"MA for {ma} days"
    company[column_name] = company['Adj Close'].rolling(ma).mean()

plt.figure(figsize=(15, 10))
company[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
plt.title('STARBUCKS')
plt.ylabel('Price')
plt.xlabel(None)
plt.legend()
plt.tight_layout()
plt.show()

# We'll use pct_change to find the percent change for each day
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
#fig, axes = plt.subplots(nrows=2, ncols=2)
plt.figure()
fig.set_figheight(10)
fig.set_figwidth(15)

SBUX['Daily Return'].plot( legend=True, linestyle='--', marker='o')
plt.title('STARBUCKS')
fig.tight_layout()

#import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
company['Daily Return'].hist(bins=50)
plt.xlabel('Daily Return')
plt.ylabel('Counts')
plt.title('STARBUCKS')

plt.tight_layout()
plt.show()

# Grab all the closing prices for the tech stock list into one DataFrame
# tech_list=['SBUX']
closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']
#closing_=closing_list# Grab all the closing prices for the tech stock list into one DataFrame

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
pd.DataFrame(tech_rets).head()
# Make a new tech returns DataFrame
#tech_rets = closing_df.pct_change()
#tech_rets.head()

sns.jointplot( data=tech_rets, kind='scatter', color='seagreen')

# sns.jointplot(x=, y=, data=tech_rets, kind='scatter', color='seagreen')

import yfinance as yf
import datetime

tech_list = ['SBUX']
start = datetime.datetime(2022, 1, 1)
end = datetime.datetime(2022, 12, 31)

data = yf.download(tech_list, start=start, end=end)
closing_df = data['Adj Close']
tech_rets = closing_df.pct_change()
tech_rets.head()

# Get the stock quote
df = pdr.get_data_yahoo('SBUX', start='2012-01-01', end=datetime.now())
# Show teh data
df

plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predicted prices
valid