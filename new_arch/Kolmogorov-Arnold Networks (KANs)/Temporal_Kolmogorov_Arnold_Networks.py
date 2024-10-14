## From https://medium.com/@brunoynobruno/unveiling-temporal-kolmogorov-arnold-networks-tkan-journey-into-advanced-time-forecasting-2cf7c272965b

"""
The article explores the integration of nonlinear dynamics and machine learning to improve temporal data forecasting, 
specifically by introducing Temporal Kolmogorov-Arnold Networks (TKAN). These novel neural networks leverage both Kolmogorov-Arnold representation 
and recurrent architectures like LSTMs to handle complex time-series data.

1. Time Series Forecasting and Nonlinearity
   Time series forecasting involves predicting future values based on past data, a common challenge in various fields like finance,
   climate science, and economics. The complexity arises from the nonlinear, stochastic, and long-term dependencies in temporal data. 
   For instance, predicting stock market movements is difficult due to the high level of randomness and volatility.

2. Kolmogorov-Arnold Networks (KAN)
   KAN is a neural network architecture inspired by the Kolmogorov-Arnold Representation Theorem, 
   which states that any multivariate function can be represented as a sum and composition of univariate functions. 
   This theorem is key to KAN’s design, allowing it to break down complex multivariate functions into simpler univariate ones. 
   This decomposition simplifies the learning process, making it more tractable to approximate highly complex functions.

   -1. B-Spline Parametrization: In KANs, univariate functions are modeled using B-splines, 
                                 which are piecewise polynomials. The B-spline representation provides flexibility and smoothness in approximating nonlinear functions.
   -2. KAN Layer Structure: Each KAN layer performs function composition based on univariate transformations. 
                            These layers are stacked to allow the network to model more complex relationships between input features.

3. Temporal Kolmogorov-Arnold Networks (TKAN)
   TKANs extend the KAN architecture to handle sequential data or time series by incorporating Recurrent Kolmogorov-Arnold Networks (RKANs) and LSTM-like memory mechanisms

   -1. RKAN Layers: Similar to RNNs, RKAN layers include a time component, maintaining short-term memory of past states in the network. 
                    The transformation functions in RKAN layers depend on time, allowing the network to handle sequential dependencies.
   -2. LSTM Memory Mechanism: TKAN integrates LSTM’s gated mechanisms to manage information flow over time. 
                              The forget gate, input gate, and output gate decide which information to retain or discard, 
                              enhancing the model’s ability to capture long-term dependencies in time series data.

   This combination of KAN’s functional decomposition with RNN and LSTM capabilities enables TKAN to excel at multi-step time series forecasting.

4. Applications and Multistep Forecasting
   The study highlights multi-step forecasting of the S&P 500 index using both TKAN and LSTM networks. 
   In multistep forecasting, models predict several future time steps instead of a single next value, 
   which increases uncertainty due to compounding errors over time.

   Mean Squared Error (MSE) and R² Score are used to evaluate the models’ accuracy.
   While MSE quantifies the average squared difference between actual and predicted values, 
   the R² score measures the proportion of variance explained by the model, with values closer to 1 indicating better performance.

5. Comparison of TKAN and LSTM
   By comparing TKAN with LSTM on different forecasting horizons (e.g., 1, 2, 3, 5, 10 steps ahead), 
   the study aims to assess the accuracy and generalization capabilities of each model. 
   This helps determine which approach better captures the underlying patterns in time series data like stock market trends.

   - TKAN Strengths: TKAN’s ability to combine RKAN’s functional representation and LSTM’s memory management offers
                     a robust framework for dealing with nonlinearities and long-term dependencies in time series. 
   - LSTM Strengths: LSTMs have established themselves as reliable models for handling temporal dependencies, 
                     but TKAN potentially provides a more efficient and accurate alternative by leveraging the univariate function decomposition from KAN.

6. Conclusion
   TKANs, by combining Kolmogorov-Arnold function representation with recurrent and memory mechanisms, present a powerful tool for time series forecasting. 
   This new approach simplifies the learning of complex temporal dependencies and offers better control over nonlinear dynamics in data, 
   providing a potentially more accurate alternative to traditional models like LSTMs, particularly for multi-step forecasting tasks.

In summary, TKAN is a promising architecture that blends theoretical insights from function theory with practical machine learning techniques to master temporal data
, opening new avenues for improving prediction in fields like finance, climate modeling, and beyond.
"""
### Example code
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tkan import TKAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


data = pd.read_csv('sp500.csv')
data['Último'] = data['Último'].apply(lambda x: float(str(x).replace(',', '')) * 1000)
data['Fecha'] = pd.to_datetime(data['Fecha'])  
data = data.sort_values(by='Fecha', ascending=True).reset_index(drop=True)

data_series = data['Último'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_series_scaled = scaler.fit_transform(data_series)

window_size = 20
future_steps_list = [1, 2, 3, 5, 6, 7, 10, 12]
results = []

for future_steps in future_steps_list:
    X, y = [], []
    for i in range(len(data_series_scaled) - window_size - future_steps + 1):
        X.append(data_series_scaled[i:i + window_size])
        y.append(data_series_scaled[i + window_size:i + window_size + future_steps].flatten())


X = np.array(X)
y = np.array(y)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
fechas_train = data['Fecha'].iloc[window_size:split_index + window_size].values
fechas_test = data['Fecha'].iloc[split_index + window_size:split_index + window_size + len(y_test)].values

model_tkan = Sequential([
    TKAN(200, sub_kan_configs=[{'spline_order': 4, 'grid_size': 12}, {'spline_order': 3, 'grid_size': 10}, {'spline_order': 5, 'grid_size': 8}], 
        return_sequences=False, use_bias=True),
    Dense(units=future_steps, activation='linear')
])

model_tkan.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history_tkan = model_tkan.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred_tkan = model_tkan.predict(X_test)
y_test_inverse = scaler.inverse_transform(y_test)
y_pred_tkan_inverse = scaler.inverse_transform(y_pred_tkan)

############
model_lstm = Sequential([
    LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(units=future_steps, activation='linear')
])

model_lstm.compile(optimizer='adam', loss='mse')
history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred_lstm = model_lstm.predict(X_test)
y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)

##############
mse_tkan = mean_squared_error(y_test_inverse, y_pred_tkan_inverse)
r2_tkan = r2_score(y_test_inverse, y_pred_tkan_inverse)

mse_lstm = mean_squared_error(y_test_inverse, y_pred_lstm_inverse)
r2_lstm = r2_score(y_test_inverse, y_pred_lstm_inverse)

results.append({
    'future_steps': future_steps,
    'R^2 TKAN': r2_tkan,
    'R^2 LSTM': r2_lstm
})

results_df = pd.DataFrame(results)









"""
