import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from data import fetch_data

# Fungsi untuk membangun model LSTM
def build_lstm_model(neurons=1, activation='relu', learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(neurons, activation=activation, input_shape=(1, 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Memuat data harga saham close BMRI
data = fetch_data()  # Ubah dengan nama file yang sesuai

# Forward fill missing date values
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data = data.set_index('date')
data = data.resample('D').ffill()

close_prices = data['close'].values.reshape(-1, 1)

# Melakukan penskalaan fitur
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Membagi data menjadi data latih dan data validasi
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
validation_data = scaled_prices[train_size:]

# Membentuk dataset dengan timesteps = 1
def create_dataset(data):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_val, y_val = create_dataset(validation_data)

# Menentukan parameter yang akan diuji
neurons = [50, 100, 150]
activations = ['relu', 'tanh', 'sigmoid']
learning_rate = [0.001, 0.01, 0.1]

# Membangun dan melatih model menggunakan GridSearchCV
model = KerasRegressor(build_fn=build_lstm_model, verbose=0)
param_grid = {'neurons': neurons, 'activation': activations, 'learning_rate': learning_rate}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)
grid_result = grid_search.fit(X_train, y_train, epochs=100)

# Memasukkan hasil ke dalam file CSV
results = grid_result.cv_results_
df_results = pd.DataFrame.from_dict(results)

# Fungsi untuk menghitung RMSE
def calculate_rmse(row):
    return np.sqrt(-row['mean_test_score'])

# Menambahkan kolom "RMSE" dengan nilai RMSE pada setiap baris
df_results['RMSE'] = df_results.apply(calculate_rmse, axis=1)

# Menyimpan DataFrame ke dalam file CSV
df_results.to_csv('LSTM_tuning_result.csv', index=False)
