# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:37:31 2020

@author: Jorge Caiado
"""

# see https://www.kaggle.com/datasets/akioonodera/monthly-temperature-of-aomori-city

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Step 1) Load CSV File into Dataframe
sns.set(rc={'figure.figsize':(15,6)})
url = 'C:/Users/jcaia/Dropbox/Jorge Caiado/FORMAÇÃO/Python/Time Series Forecasting with Python/Data/monthly_temperature_aomori_city.csv'
df = pd.read_csv(url)
print(df.info())

# Step 2) Create the column Date (dd/mm/yyyy) and set it as index
df['DATE'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
df.set_index('DATE',inplace=True)
print(df.head())

# Step 3) Plot the data to find patterns
plt.figure(figsize=(15,6))
plt.plot(df.temperature)
plt.plot(df.temperature[1656-60:]) #last 60 obs. 

# Step 4) Time Series decomposition
decomposition = sm.tsa.seasonal_decompose(df.temperature, model='additive')
plt.rcParams["figure.figsize"] = [16,9]
fig = decomposition.plot()

# Step 5) Split data into train and test (last 36 obs.) samples
train = df.temperature[:1656-36]
test = df.temperature[1656-36:]

# Step 5) Check data stationary: ADF unit root test
from statsmodels.tsa.stattools import adfuller
def check_stationarity(timeseries):    
    result = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(result[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
check_stationarity(train)

# Apply first differences and check stationarity
np.diff(train)
df_diff = np.diff(train)
plt.plot(df_diff)
check_stationarity(df_diff)

# Apply seasonal differences and check stationarity
df['dtemp12'] = df['temperature'] - df['temperature'].shift(12)
df_diff12 = df['dtemp365'].to_numpy()
df_diff12 = df_diff12[~np.isnan(df_diff12)] #drop missing data
plt.plot(df_diff12)
check_stationarity(df_diff12)

# Apply seasonal differences and nonseasonal differences and check stationarity
df['ddtemp12'] = df['dtemp12'] - df['dtemp12'].shift(1)
df_ddiff12 = df['ddtemp12'].to_numpy()
df_ddiff12 = df_ddiff12[~np.isnan(df_ddiff12)] #drop missing data
plt.plot(df_ddiff12)
check_stationarity(df_ddiff12)


# Plot ACF and PACF of the original series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(10,2))
plot_acf(train, lags=36)
plot_pacf(train, lags=36)

# Plot ACF and PACF of the differenced series ddiff12
plt.figure(figsize=(10,2))
plot_acf(df_ddiff12, lags=36)
plot_pacf(df_ddiff12, lags=36)

#################################
# Step 6) Fit SARMA/SARIMA models

# Fit a SARIMA(0,1,1)(0,1,1)12 model
model1 = sm.tsa.statespace.SARIMAX(train, 
                                order=(0,1,1), 
                                seasonal_order=(0,1,1,12)
                                )
results1 = model1.fit()
print(results1.summary())
# Plot residual errors
resid1 = pd.DataFrame(results1.resid)
fig, axes = plt.subplots()  
axes.plot(resid1)
axes.set_title("Residuals")
fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(resid1.dropna(), lags=36, ax=axes[0])
plot_pacf(resid1.dropna(), lags=36, ax=axes[1])
plt.show()


# Fit a SARIMA(1,1,1)(0,1,1)12 model
model2 = sm.tsa.statespace.SARIMAX(train, 
                                order=(1,1,1), 
                                seasonal_order=(0,1,1,12)
                                )
results2 = model2.fit()
print(results2.summary())
# Plot residual errors
resid2 = pd.DataFrame(results2.resid)
fig, axes = plt.subplots()  
axes.plot(resid2)
axes.set_title("Residuals")
fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(resid2.dropna(), lags=36, ax=axes[0])
plot_pacf(resid2.dropna(), lags=36, ax=axes[1])
plt.show()

# Step 6) Predict the last 36 observations using the best SARMA/SARIMA models
df['forecast'] = results2.predict(start=1656-36, end=1656, dynamic= True)  
df[['temperature', 'forecast']][1656-72:].plot(figsize=(12, 8)) 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_absolute_error

test = df.iloc[-36:] #last 12 observations

# Forecast Accuracy Measures
rmse = round(np.sqrt(mean_squared_error(test['temperature'],test['forecast'])),2)
print("Root Mean Squared Error (RMSE) =",rmse)

mae = round(mean_absolute_error(test['temperature'],test['forecast']),2)
print("Mean Absolute Error (MAE) =",mae)

mape = round(100*mean_absolute_percentage_error(test['temperature'],test['forecast']),2)
print("Mean Absolute Percentual Error (MAPE) =",mape,'%')

maen = round(mean_absolute_error(test['temperature'][1:],test['temperature'][:-1]),2)
print("Mean Absolute Error for Naive Forecast (MAEN) =",maen)




