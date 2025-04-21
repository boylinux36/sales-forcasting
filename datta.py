
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load your sales data (adjust the path)
df = pd.read_csv('sales_data_sample.csv', parse_dates=['ORDERDATE'], index_col='ORDERDATE',encoding="ISO-8859-1")

# Resample to monthly data (if the data is daily)
monthly_sales = df['SALES'].resample('M').sum()

# Fit ARIMA model (tune parameters if necessary)
model = ARIMA(monthly_sales, order=(5, 1, 0))
model_fit = model.fit()

# Forecast for the next 12 months
forecast = model_fit.forecast(steps=12)
forecast_dates = pd.date_range(monthly_sales.index[-1], periods=13, freq='M')[1:]

# Create subplots to display all the figures in one output
fig, axs = plt.subplots(3, 1, figsize=(5, 10))

# Plot 1: Historical Sales Data
axs[0].plot(monthly_sales.index, monthly_sales, label='Historical Sales')
axs[0].set_title('Historical Sales Data')
axs[0].set_xlabel('ORDERDATE')
axs[0].set_ylabel('SALES')
axs[0].legend()

# Plot 2: Monthly Sales Data (in case it's resampled or processed differently)
monthly_sales.plot(ax=axs[1], color='tab:blue')
axs[1].set_title('Monthly Sales Data')
axs[1].set_xlabel('ORDERDATE')
axs[1].set_ylabel('SALES')

# Plot 3: Forecasted Sales Data
axs[2].plot(monthly_sales.index, monthly_sales, label='Historical Sales')
axs[2].plot(forecast_dates, forecast, label='Forecasted Sales', color='red')
axs[2].set_title('Sales Forecast for the Next Year')
axs[2].set_xlabel('ORDERDATE')
axs[2].set_ylabel('SALES')
axs[2].legend()

# Display the plots
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# Optionally, if you want to evaluate the model with actual data:
# test_data = pd.read_csv('test_sales_data.csv', parse_dates=['Date'], index_col='Date')
# test_sales = test_data['Sales'].resample('M').sum()

# mae = mean_absolute_error(test_sales, forecast)
# rmse = np.sqrt(mean_squared_error(test_sales, forecast))

# print(f'Mean Absolute Error: {mae}')
# print(f'Root Mean Squared Error: {rmse}')
