#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: University of Sydney Business School
         Discipline of Business Analytics
"""

# Import libraries 
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow           # Install: pip install drawnow
import time

N = 200                               # Number of observations
t = np.arange(1,201,1) /200.0
y = 0.6*t                             # Make a simulation time series


s = np.random.randint(1,7,10)        # Generate random integer numbers
s = s * np.array([1, -1, 1, 1, -1, -1, 1, 1, 1, -1])
s = np.tile(s, 20)                   # Replicate array s 20 times

# The function to draw the pure trend
def draw_trend():
    plt.plot(t,y)
    plt.title('The Trend')

# The function to draw the trend plus seasonal
def draw_seasonal():
    plt.plot(y+s/50.0)
    plt.title('The Trend Plus Seasonals')
    
# The function to draw the simulated data
def draw_withNoise():
    plt.plot(y+s/50.0+0.05*np.random.normal(size=200))
    plt.title('A Synthetic Possibly True Time Series')

# The function to draw the seasonal adjusted data 
def draw_adjusted(): 
    plt.plot(y+0.01*np.random.normal(size=200))
    plt.title('Seasonally Adjusted Data!')

time.sleep(3)                # Sleep 3 seconds
drawnow(draw_trend)          # Call the function to plot the pure trend
time.sleep(5)
drawnow(draw_trend)
time.sleep(5)
drawnow(draw_seasonal)       # Call the function to plot trend + seasonal
time.sleep(20)
drawnow(draw_seasonal)
time.sleep(5)
drawnow(draw_withNoise)      # Call the function to plot trend + seasonal + fluctuation
time.sleep(5)
drawnow(draw_withNoise)
time.sleep(5)
drawnow(draw_adjusted)       # Call the function to plot the seasonal adjusted data

