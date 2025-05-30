import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('daily_temperature.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

df_prophet = df.rename(columns={'Date': 'ds', 'Temperature': 'y'})

df_prophet = df_prophet[df_prophet['City'] == 'New York']

model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

df_prophet.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)
df_merged = df_prophet.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='inner')

df_merged.reset_index(inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df_merged['ds'], df_merged['y'], 'b-', label='Actual Temperature', marker='o', markersize=8)
plt.plot(df_merged['ds'], df_merged['yhat'], 'r-', label='Predicted Temperature', marker='o',markersize=8)

for i, txt in enumerate(df_merged['y']):
    plt.annotate(round(txt, 2), (df_merged['ds'][i], df_merged['y'][i]), textcoords="offset points", xytext=(0,10), ha='center')

for i, txt in enumerate(df_merged['yhat']):
    plt.annotate(round(txt, 2), (df_merged['ds'][i], df_merged['yhat'][i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error
y_test = df_merged['y'].values
predictions = df_merged['yhat'].values

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Aboulute Error: {mae}")