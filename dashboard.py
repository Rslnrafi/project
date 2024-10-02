import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import seaborn as sns
import streamlit as st
import pickle

#load model
proyek_akhir = pickle.load(open('bike_sharing.sav', 'rb'))


#title
st.title('Proyek Akhir Belajar Analisis dengan Python')


# Load datasets
day_data = pd.read_csv('E:\MSIB\Bangkit\Belajar Analisis Data Dengan Python\day.csv')
day_data.head()
hour_data = pd.read_csv('E:\MSIB\Bangkit\Belajar Analisis Data Dengan Python\hour.csv')
hour_data.head()

# Prepare data for regression model (day.csv)
X_day = day_data[['temp', 'hum', 'windspeed']]
y_day = day_data['cnt']

# Add a constant (intercept) to the model
X_day = sm.add_constant(X_day)

# Fit the regression model
model_day = sm.OLS(y_day, X_day).fit()

# Print the summary of the regression model
print(model_day.summary())

# Group by season, month, and weekday to observe trends
season_trend = day_data.groupby('season')['cnt'].mean()
month_trend = day_data.groupby('mnth')['cnt'].mean()
weekday_trend = day_data.groupby('weekday')['cnt'].mean()

# Plot the trends
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Season Trend
axes[0].bar(season_trend.index, season_trend.values, color='lightblue')
axes[0].set_title('Average Bike Usage by Season')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])

# Month Trend
axes[1].bar(month_trend.index, month_trend.values, color='lightgreen')
axes[1].set_title('Average Bike Usage by Month')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Weekday Trend
axes[2].bar(weekday_trend.index, weekday_trend.values, color='lightcoral')
axes[2].set_title('Average Bike Usage by Weekday')
axes[2].set_xticks(range(7))
axes[2].set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

plt.tight_layout()
plt.show()