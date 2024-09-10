#!/usr/bin/env python
# coding: utf-8

# # Here's a step-by-step guide to creating a financial market data analysis database in Anaconda's Jupyter Notebook using Python:
# 
# Step 1: Install necessary libraries
# 
# Open Anaconda's Jupyter Notebook and create a new notebook.
# Install the following libraries by running the following commands

# pip install pandas
# pip install yfinance
# pip install mplfinance
# pip install scipy
# pip install statsmodels

#  Step 2: Import necessary libraries

# In your Jupyter Notebook, import the necessary libraries:

# In[5]:


get_ipython().system('pip install mplfinance')


# Check if installation is successful:
# After installing mplfinance, you can check if it's installed correctly by running:

# In[6]:


import mplfinance as mpf
print(mpf.__version__)


# In[7]:


import pandas as pd
import yfinance as yf
import mplfinance as mpf
from scipy.stats import norm
import statsmodels.api as sm


# # Step 3: Download historical financial market data
# 
# Use the yfinance library to download historical financial market data for a specific stock or index. For example:

# In[9]:


data = yf.download('AAPL', start='2020-01-01', end='2022-02-26')


# This code downloads the historical stock price data for Apple (AAPL) from January 1, 2020 to February 26, 2022.
# 
# Step 4: Clean and preprocess the data
# 
# Convert the downloaded data into a Pandas DataFrame:

# In[10]:


df = pd.DataFrame(data)


# Clean and preprocess the data by handling missing values, converting dates to datetime format, and setting the index to the date column:

# In[14]:


import pandas as pd
df = pd.DataFrame(data)
print(df.isnull().sum())


# In[15]:


df = df.dropna()  # drop missing values
df['Date'] = pd.to_datetime(df.index)  # convert index to datetime format
df.set_index('Date', inplace=True)  # set index to date column


# Step 5: Analyze and visualize the data
# 
# Use various statistical and visualization techniques to analyze and visualize the financial market data. For example:

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Assuming df is your DataFrame and 'Returns' is your column name

# Fit a Gaussian distribution to the returns
df['Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
dist = norm.fit(df['Returns'].dropna())

# Plot the distribution of returns
plt.hist(df['Returns'].dropna(), bins=50)
plt.title('Distribution of Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()


# In[23]:


# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Calculate daily volatility
df['Volatility'] = df['Returns'].rolling(window=20).std()

# Plot the stock price and volatility
mpf.plot(df, type='candle', style='yahoo', volume=True, title='AAPL Stock Price and Volatility')

# Calculate statistical metrics (e.g. mean, std, skewness)
print(df.describe())

# Plot the distribution of returns
import matplotlib.pyplot as plt
plt.hist(df['Returns'], bins=50, density=True)
plt.show()


# In[ ]:




