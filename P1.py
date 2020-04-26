import warnings
import statsmodels.api as sm
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

sns.set(font='IPAGothic')
# specify training data
data = pd.read_csv("temp.csv")

# define model configuration
my_order = (1, 1, 1)
my_seasonal_order = (1, 1, 1, 12)
# define model
model = sm.tsa.statespace.SARIMAX(data, order=my_order, seasonal_order=my_seasonal_order)
# fit model
model_fit = model.fit()
# one step forecast
yhat = model_fit.forecast()
yhat2 = model_fit.predict(start=len(data), end=len(data))
