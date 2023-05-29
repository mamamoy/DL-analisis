import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam

# Fungsi untuk membangun model RNN
def build_rnn_model(neurons=1, activation='relu', learning_rate=0.001):
    model = Sequential()
    model.add(SimpleRNN(neurons, activation=activation, input_shape=(1, 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Memuat data harga saham close BMRI
data = pd.read_csv('BMRI.JK.csv')  # Ubah dengan nama file yang sesuai

# Forward fill missing date values
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.set_index('Date')
data = data.resample('D').ffill()

close_prices = data['Close'].values.reshape(-1, 1)

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
model = KerasRegressor(build_fn=build_rnn_model, verbose=0)
param_grid = {'neurons': neurons, 'activation': activations, 'learning_rate': learning_rate}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)
grid_result = grid_search.fit(X_train, y_train, epochs=100)

# Menampilkan hasil grid search
print("Grid Search Results:")
print()
print("Best Parameters: ", grid_result.best_params_)
print("Best RMSE Score: ", -grid_result.best_score_)
print()

# Menampilkan nilai parameter dan nilai RMSE setiap kali melakukan percobaan pada GridSearchCV
results = grid_result.cv_results_
for i in range(len(results['params'])):
    print("Parameters: ", results['params'][i])
    print("RMSE Score: ", -results['mean_test_score'][i])
    print()
