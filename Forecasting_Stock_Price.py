import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing , svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Stock:
    one_day = 86400

    def __init__(self, stock_name, path):
        self.name = stock_name
        self.path = path

        # Reading data from csv file
        self.df = pd.read_csv(self.path)

        # Update date column's data type with timestamp
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Set Date column as dataframe index
        self.df = self.df.set_index('Date')

        # Pick necessary columns
        self.df = self.df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

        # Calculate percentage of changes
        self.df['HL PCT'] = (self.df['High'] - self.df['Adj Close']) / (self.df['Adj Close']) * 100
        self.df['PCT_change'] = (self.df['Adj Close'] - self.df['Open']) / (self.df['Open']) * 100
        # self.df = self.df[['Adj Close', 'HL PCT', 'PCT_change', 'Volume']]

        # Calculate default forecast value
        self.forecast_day = int(math.ceil(0.02 * len(self.df)))

    def forecasting(self, day=None, column='Adj Close'):
        # Check day value
        if day is None:
            day = self.forecast_day

        if not column in self.df.columns:
            print("This column is not exist!")
            return {}

        # Prepare Dataframe for forecasting
        # Fill holes with -9999
        self.df.fillna(-9999, inplace=True)

        # Move forecasting column's data to new column by shifting forecasting row count
        self.df['label'] = self.df[column].shift(-day)
        x = np.array(self.df.drop(['label'], 1))
        x = preprocessing.scale(x)
        x = x[:-day]
        x_lately = x[-day:]

        self.df.dropna(inplace=True)
        y = np.array(self.df['label'])
        y = np.array(self.df['label'])

        # Split dataframe as train and test data for x and y axis
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # Run Regression method over data
        clf = LinearRegression(n_jobs=-1)
        # clf = svm.SVR()
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)

        forecast_set = clf.predict(x_lately)
        self.df['Forecast'] = np.nan

        # Get maximum date value
        last_date = self.df.iloc[-1].name
        last_unix = last_date.timestamp()
        next_unix = last_unix + Stock.one_day

        # Append predicted values to  DataFrame
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += Stock.one_day
            self.df.loc[next_date] = [np.nan for _ in range(len(self.df.columns) - 1)] + [i]

        return {'values': forecast_set, 'accuracy': accuracy, 'days': day}

    def plot(self, label='Price', col='Adj Close', forecast=False):
        style.use('fivethirtyeight')
        self.df[col].plot()
        if forecast:
            self.df['Forecast'].plot()

        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel(label)
        plt.show()

    def get_columns(self):
        return self.df.columns

    def summary(self, col='Adj Close'):
        maximum = np.max(self.df[col])
        minimum = np.min(self.df[col])

        max_day = self.df[self.df[col] == maximum].index[0].strftime("%Y-%m-%d")
        min_day = self.df[self.df[col] == minimum].index[0].strftime("%Y-%m-%d")

        return maximum, max_day, minimum, min_day


if __name__ == "__main__":
    stock = Stock("FB", "/Users/merveozmen/PycharmProjects/untitled5/FB.csv")
    # print(stock.get_columns()[0])
    print(stock.forecasting(30,'PCT_change'))
    # stock.plot(label='Forecast', col='Forecast', forecast=False)
    stock.plot(label='Open', col=stock.get_columns()[0], forecast=False)
    stock.plot(label='PCT_change', col='PCT_change', forecast=True)
    # print(stock.summary())
