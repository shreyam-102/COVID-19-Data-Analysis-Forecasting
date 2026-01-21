# COVID-19 Data Analysis & Forecasting
# Author: Project Implementation
# Description: Analysis and forecasting of COVID-19 cases (Global & India)

import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
# NOTE: Replace file paths when adding datasets
global_df = pd.read_csv("data/global_covid_data.csv")
india_df = pd.read_csv("data/india_covid_data.csv")

# Convert Date column to datetime
global_df['Date'] = pd.to_datetime(global_df['Date'])
india_df['Date'] = pd.to_datetime(india_df['Date'])

# -------------------------------
# STEP 2: Daily New Cases
# -------------------------------
global_df['NewConfirmed'] = global_df['Confirmed'].diff()
global_df['NewDeaths'] = global_df['Deaths'].diff()
global_df['NewRecovered'] = global_df['Recovered'].diff()

india_df['NewConfirmed'] = india_df['Confirmed'].diff()
india_df['NewDeaths'] = india_df['Deaths'].diff()
india_df['NewRecovered'] = india_df['Recovered'].diff()

# -------------------------------
# STEP 3: Rolling Averages & Rates
# -------------------------------
global_df['Confirmed_7d'] = global_df['NewConfirmed'].rolling(7).mean()
global_df['CFR'] = global_df['Deaths'] / global_df['Confirmed']
global_df['RecoveryRate'] = global_df['Recovered'] / global_df['Confirmed']

india_df['Confirmed_7d'] = india_df['NewConfirmed'].rolling(7).mean()
india_df['CFR'] = india_df['Deaths'] / india_df['Confirmed']
india_df['RecoveryRate'] = india_df['Recovered'] / india_df['Confirmed']

# -------------------------------
# STEP 4: Visualization (Plotly)
# -------------------------------
fig_global = px.line(
    global_df,
    x='Date',
    y=['Confirmed', 'Recovered', 'Deaths'],
    title='Global COVID-19 Trends'
)
fig_global.show()

fig_india = px.line(
    india_df,
    x='Date',
    y=['Confirmed', 'Recovered', 'Deaths'],
    title='India COVID-19 Trends'
)
fig_india.show()

# -------------------------------
# STEP 5: Forecasting using Prophet
# -------------------------------
forecast_df = india_df[['Date', 'Confirmed']].rename(
    columns={'Date': 'ds', 'Confirmed': 'y'}
)

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# -------------------------------
# STEP 6: Forecast Visualization
# -------------------------------
model.plot(forecast)
model.plot_components(forecast)

print("Forecasting completed successfully.")
