# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:44:06 2020

@author: Jorge Caiado
"""

# Monthly count of riders for the Portland public transportation system

# Source: https://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf


url = "https://raw.githubusercontent.com/PinkWink/DataScience/master/data/07.%20portland-oregon-average-monthly-.csv" 
df = pd.read_csv(url, index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.columns= ['riders']
print(df)

# Plot data
df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)

# Compute Sample ACF and Sample PACF
fig0 = plt.figure(figsize=(12,8))
ax1 = fig0.add_subplot(211)
fig0 = sm.graphics.tsa.plot_acf(df.riders, lags=36, ax=ax1)
ax2 = fig0.add_subplot(212)
fig0 = sm.graphics.tsa.plot_pacf(df.riders, lags=36, ax=ax2)

# Unit root tests
def test_stationarity(timeseries):
       
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value',
                                '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
        
test_stationarity(df.riders)  

# Log of Riders
df.riders_log= df.riders.apply(lambda x: np.log(x))  
test_stationarity(df.riders_log)
df.riders_log.plot(figsize=(12,8), title= 'Logarithm of riders', fontsize=14)

# First differences
df['first_difference'] = df.riders - df.riders.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))
df['first_difference'].plot(figsize=(12,8), 
                            title= 'First difference', fontsize=14)

# Differences of Log
df['log_first_difference'] = df.riders_log - df.riders_log.shift(1)  
test_stationarity(df.log_first_difference.dropna(inplace=False))


# Sesaonal Differences
df['seasonal_difference'] = df.riders - df.riders.shift(12)  
test_stationarity(df.seasonal_difference.dropna(inplace=False))
df['seasonal_difference'].plot(figsize=(12,8), 
                               title= 'Sesaonal difference', fontsize=14)


# Ordinary and sesaonal differences
df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(12)  
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))
df['seasonal_first_difference'].plot(figsize=(12,8), 
                                     title= 'Ordinary and seasonal differences', fontsize=14)


# Ordinary and sesaonal differences of logs
df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(12)  
test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))
df['log_seasonal_first_difference'].plot(figsize=(12,8), 
                                     title= 'Ordinary and seasonal differences of logarithm of riders', fontsize=14)



#####################
# ACF and PACF of stationary time series: Ordinary and sesaonal differences
# Do not include the first 13 observations
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[13:], 
                               lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[13:],
                                lags=40, ax=ax2)

###########################
# Fit a SARIMA(0,1,0)(0,1,1)12 model
model1 = sm.tsa.statespace.SARIMAX(df.riders, trend='n', 
                                order=(0,1,0), 
                                seasonal_order=(0,1,1,12))
results1 = model1.fit()
print(results1.summary())

#########################
# Fit a SARIMA(0,1,0)(0,1,1)12 model
model2 = sm.tsa.statespace.SARIMAX(df.riders, trend='n', 
                                order=(0,1,0), 
                                seasonal_order=(1,1,1,12))
results2 = model2.fit()
print(results2.summary())

#################
# Forecast the last 12 obs. using model2
df['forecast'] = results2.predict(start = 102, end= 114, dynamic= True)  
df[['riders', 'forecast']].plot(figsize=(12, 8)) 

# Compute forecast accuarcy measures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_absolute_error

test = df.iloc[-12:] #last 12 observations

# Forecast Accuracy Measures
rmse_riders = round(np.sqrt(mean_squared_error(test['riders'],test['forecast'])),2)
print("Root Mean Squared Error (RMSE) =",rmse_riders)

mae_riders = round(mean_absolute_error(test['riders'],test['forecast']),2)
print("Mean Absolute Error (MAE) =",mae_riders)

mape_riders = round(100*mean_absolute_percentage_error(test['riders'],test['forecast']),2)
print("Mean Absolute Percentual Error (MAPE) =",mape_riders,'%')

maen_riders = round(mean_absolute_error(test['riders'][1:],test['riders'][:-1]),2)
print("Mean Absolute Error for Naive Forecast (MAEN) =",maen_riders)
 







  