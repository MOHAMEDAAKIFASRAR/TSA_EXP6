# Ex.No: 6               HOLT WINTERS METHOD
### Date: 21-11-2026
### DataSet Name : AirPassengers

### AIM:
To analyze monthly air passenger data and forecast future passenger numbers using the Holt-Winters Exponential Smoothing Method.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("AirPassengers.csv")

print(data.head())
print(data.info())

data['Month'] = pd.to_datetime(
    data['Month'],
    format='%Y-%m'
)
data.set_index('Month', inplace=True)
data.sort_index(inplace=True)

print(data.head())
data_monthly = data['#Passengers'].dropna()

print("\nMonthly Passenger Data:")
print(data_monthly.head())

print("\nNumber of available observations:")
print(len(data_monthly))

plt.figure(figsize=(12, 5))

plt.plot(
    data_monthly,
    label='Passengers'
)

plt.title('Monthly Air Passengers')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# SEASONAL DECOMPOSITION
# ---------------------------------------------------

decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

decomposition.plot()

plt.suptitle(
    "Seasonal Decomposition of Air Passengers",
    y=1.02
)

plt.show()


# ---------------------------------------------------
# SPLIT TRAIN AND TEST DATA
# ---------------------------------------------------

train_size = int(len(data_monthly) * 0.8)

train_data = data_monthly.iloc[:train_size]
test_data = data_monthly.iloc[train_size:]

print("Training data size:", len(train_data))
print("Testing data size:", len(test_data))


# ---------------------------------------------------
# CREATE HOLT-WINTERS MODEL
# ---------------------------------------------------

model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# ---------------------------------------------------
# PREDICT TEST DATA
# ---------------------------------------------------

test_predictions = model.forecast(
    steps=len(test_data)
)


# ---------------------------------------------------
# VISUAL EVALUATION
# ---------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    train_data,
    label='Training Data'
)

plt.plot(
    test_data,
    label='Actual Test Data'
)

plt.plot(
    test_predictions,
    label='Predicted Test Data'
)

plt.title('Holt-Winters Model - Air Passenger Prediction')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')

plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# MODEL PERFORMANCE METRICS
# ---------------------------------------------------

mae = mean_absolute_error(
    test_data,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions
    )
)

print("MAE:", mae)
print("RMSE:", rmse)


# ---------------------------------------------------
# CREATE FINAL MODEL USING ALL AVAILABLE DATA
# ---------------------------------------------------

final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# ---------------------------------------------------
# PREDICT FUTURE DATA
# ---------------------------------------------------

future_steps = 32

final_predictions = final_model.forecast(
    steps=future_steps
)


# ---------------------------------------------------
# PLOT FINAL PREDICTION
# ---------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data_monthly,
    label='Actual Passengers'
)

plt.plot(
    final_predictions,
    label='Future Prediction'
)

plt.title('Holt-Winters Air Passenger Forecast')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')

plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# DISPLAY FUTURE PREDICTIONS
# ---------------------------------------------------

future_forecast = pd.DataFrame({
    'Predicted_Passengers': final_predictions
})

print("\nFuture Passenger Predictions:")
print(future_forecast)
```


### OUTPUT:
<img width="607" height="593" alt="image" src="https://github.com/user-attachments/assets/db764f07-27e2-4a29-b345-d60cf3ddd9a6" />

<img width="960" height="485" alt="image" src="https://github.com/user-attachments/assets/14c8a05b-1d08-4b80-bf47-42d2999829c1" />

<img width="722" height="547" alt="image" src="https://github.com/user-attachments/assets/a1a2e20c-5515-4c6d-a395-63ddf5d41e5a" />

<img width="957" height="477" alt="image" src="https://github.com/user-attachments/assets/8c8c3ea0-e21b-48c5-af7c-7e7b34339ff6" />

<img width="945" height="441" alt="image" src="https://github.com/user-attachments/assets/79261366-b068-4233-8048-a285cd9b39b5" />

<img width="742" height="715" alt="image" src="https://github.com/user-attachments/assets/9bb51c70-c350-4ed4-a886-cd44d961bae4" />

### RESULT:
Thus, the Holt-Winters Method was successfully implemented to analyze the air passenger data, evaluate the model using MAE and RMSE, and predict future passenger numbers.
