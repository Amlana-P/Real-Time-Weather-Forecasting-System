import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split

# get data
data = pd.read_csv('delhidata.csv', usecols=['temp'])
"""
data = pd.read_csv('delhidata.csv', usecols=['datetime', 'temp'])
"""

# display data
"""
X_train, X_test, y_train, y_test = train_test_split(data['datetime'], data['temp'], test_size=0.2)
fig = px.line(data, x='datetime', y='temp', title='Delhi Temperature Data')
"""

# Filling missing values
median = data.median()
data.fillna(median, inplace=True)

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
mod = sm.tsa.statespace.SARIMAX(data, trend='n', order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
results = mod.fit()
"""
print(results.summary())
"""

# Prediction
# one step forecast
prediction = results.forecast()
a = list(median)[0]
b = list(prediction)[0]
accuracy = 100 - (((b - a)/a)*100)
print("Median  value : " + str(a))
print("Predicted Value : " + str(b))
print("Accuracy Percentage : " + str(accuracy))
