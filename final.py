import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# get data
data = pd.read_csv('delhidata.csv', usecols=['temp'])
"""
data = pd.read_csv('delhidata.csv', usecols=['datetime', 'temp'])
"""

# Filling missing values
median = data.median()
data.fillna(median, inplace=True)

# Convert the dataframe in to list
data_list = data.values.tolist()

# Split data
look_back = 20
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data_list)


# Reshape the list into data matrix
def create_dataset(data_set, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data_set)-look_back):
        dataX.append(data_set[i:(i+look_back), 0])
        dataY.append(data_set[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# display data
"""
X_train, X_test, y_train, y_test = train_test_split(data['datetime'], data['temp'], test_size=0.2)
fig = px.line(data, x='datetime', y='temp', title='Delhi Temperature Data')
"""

# Define and Fit LSTM Network
batch_size = 1
model = Sequential()
model.add(LSTM(32, input_dim=1))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=batch_size, verbose=2)

# Evaluation
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size, verbose=0)
print('Test Score: ', testScore)

# Dicky-Fuller Test
"""
X = data.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
"""

# ACF & P-ACF plots
"""
plot_acf(data)
plot_pacf(data, lags=50)
plt.show()
"""

# Fit SARIMA Model
# mod = sm.tsa.statespace.SARIMAX(data, trend='n', order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
# results = mod.fit()
"""
print(results.summary())
"""

# Prediction
# one step forecast
# prediction = results.forecast()
# a = list(median)[0]
# b = list(prediction)[0]
# accuracy = 100 - (((b - a)/a)*100)
# print("Median  value : " + str(a))
# print("Predicted Value : " + str(b))
# print("Accuracy Percentage : " + str(accuracy))
