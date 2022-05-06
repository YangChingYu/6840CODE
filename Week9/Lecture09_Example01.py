#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: University of Sydney Business School
         Discipline of Business Analytics
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import statsmodels.api as smt

#%% Read data
data = pd.read_csv('MonthHotel.txt',header=None)

# Seasonal period
M = 12

data_v = data[0].values
data_log = np.log(data_v)
             
#%% Plot original series and log transformation
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(data_v)
plt.title('Monthly Hotel')
ax2 = fig.add_subplot(212)
ax2.plot(data_log)
plt.title('Log of Monthly Hotel')

#%%
# Quartic root
data_qr = np.power(data_v,0.25)
data_qr_d = np.diff(data_qr)  

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(data_qr)
plt.title('Quartic Roots (QR) of Monthly Hotel')
ax2 = fig.add_subplot(212)
ax2.plot(data_qr_d)
plt.title('Differencing Quartic Roots of Monthly Hotel')

#%% Do seasonally differencing of quartic root
data_ds = data_qr[12:] - data_qr[:-12]
# Then first order difference
data_dsd = np.diff(data_ds);

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(data_ds)
plt.title('Seasonally Differenced Monthly Hotel')
ax2 = fig.add_subplot(212)
ax2.plot(data_dsd)
plt.title('Regular Difference of the Seasonal Differenced')

#%%
# Draw ACF and PACF
# For the quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_qr, lags=40, ax=ax1)
ax1.set_title("ACF: Quartic Root Data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_qr, lags=40, ax=ax2)
ax2.set_title("PACF: Quartic Root Data")

#%%
# For the ordinary difference of quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_qr_d, lags=40, ax=ax1)
ax1.set_title("ACF: first order difference of quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_qr_d, lags=40, ax=ax2)
ax2.set_title("PACF: first order difference of quartic root data")

#%%
# For the seasonal difference of quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_ds, lags=40, ax=ax1)
ax1.set_title("ACF: seasonal difference of quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_ds, lags=40, ax=ax2)  
ax2.set_title("PACF: seasonal difference of quartic root data")   

#%%
"""
Sample ACF has a spike at lag 12 and cuts off after lag 12, and the
Sample PACF dies down fairly quickly, since the spikes in this function
at lags 12 and 24 are of decreasing size. There should be a seasonal
MA(1) component.
"""
# For first order difference of seasonally differenced quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_dsd, lags=40, ax=ax1)
ax1.set_title("ACF: differencing the seasonally differenced quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_dsd, lags=40, ax=ax2)  
ax2.set_title("PACF: differencing the seasonally differenced quartic root data")

#%%
"""
We see that this sample ACF might be interpreted as dying down fairly
quickly at the nonseasonal level and cutting off fairly quickly at the
seasonl level. However this dying-down behavior at the nonseasonal level
and cutting-off behaviour at the seasonal level do not appear to be as
quick as the dying-down behaviour at the nonseasonal level and the
cutting-off behavior at the seasonal level as the last illustration
"""                      

"""
Conclusion
We will build a model based on the Seasonally differenced quartic roots
of monthly hotel

1. At the nonseasonal level the Sample PACF has spikes at lags 1, 3 and 5,
and cuts off after lag 5 and the the Sample ACF dies down. Therefore we
tentatively identify the following nonseasonal autoregressive model
Z_ds(t) = c + \phi_1 Z_ds(t-1) + \phi_3Z_ds(t-3) + \phi_5Z_ds(t-5) + e_t
 
2. At the seasonal level the sample ACF has a spike at lag 12 and cuts
off after lag 12, and Sample PACF dies down. Therefore, we tentatively
identify the seasonal moving average model of oder 1
   Z_ds(t) = c + e(t) + \Theta_1 e(t-12)

3. Combining these models, we obtain the overall tentatively identified
 model
  Z_ds(t) = c + \phi_1 Z_ds(t-1) + \phi_3Z_ds(t-3) + \phi_5Z_ds(t-5) + e(t) + \Theta_1 e(t-12)
"""

# Define the model according to the identificated pattern
# sarima_model = smt.tsa.statespace.SARIMAX(data_qr, order=(5,0,0), seasonal_order=(0,1,1,12))   
# we just say p = 5 including AR lags 1, 2, 3, 4, 5
sarima_model = smt.tsa.statespace.SARIMAX(data_log, \
                                          order=(5,0,0), 
                                          seasonal_order=(0,1,1,12))   

#%%
# Estimating the model
result = sarima_model.fit(disp=False)
print(result.summary())

#%%
# Forecasting
forecasts = result.forecast(100)

#%%
# Display forecasting
fig = plt.figure(figsize=(10,8)) 
plt.plot(data_log,label='')
plt.plot(np.arange(168,268), forecasts)
plt.show()
