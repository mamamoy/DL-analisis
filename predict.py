from data import fetch_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, GRU, LSTM
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

def predict_prices(num_days, model_name):
    def build_model(neurons=1, activation='relu', learning_rate=0.001):
        model = Sequential()
        if model_name == 'RNN':
            model.add(SimpleRNN(neurons, activation=activation, input_shape=(1, 4)))  # Input shape untuk fitur Open, High, Low, Close
        elif model_name == 'GRU':
            model.add(GRU(neurons, activation=activation, input_shape=(1, 4)))  # Input shape untuk fitur Open, High, Low, Close
        elif model_name == 'LSTM':
            model.add(LSTM(neurons, activation=activation, input_shape=(1, 4)))  # Input shape untuk fitur Open, High, Low, Close
        else:
            raise ValueError("Invalid model_name. Please choose 'SimpleRNN', 'GRU', or 'LSTM'.")
        model.add(Dense(4))  # Output layer memiliki 4 neuron sesuai dengan 4 fitur (Open, High, Low, Close)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    data = fetch_data()  # Gunakan fetch_data() sebagai data pelatihan
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.dropna().reset_index()
    data = data.set_index('Date')
    data = data.resample('D').ffill()

    selected_features = ['Open', 'High', 'Low', 'Close']
    scaled_data = {}
    for feature in selected_features:
        feature_data = data[feature].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(feature_data)

    def create_dataset(data):
        X, y = [], []
        for i in range(len(data) - 1):
            X.append(data[i])
            y.append(data[i + 1])
        return np.array(X), np.array(y)

    X, y = create_dataset(np.column_stack([scaled_data[feature] for feature in selected_features]))

    best_neuron = 150  # Contoh dari hyperparameter terbaik yang ditemukan
    best_activation = None
    if model_name == 'RNN':
        best_activation = 'tanh'
    else:
        best_activation = 'sigmoid'
    best_learning_rate = 0.001

    model = build_model(neurons=best_neuron, activation=best_activation, learning_rate=best_learning_rate)
    model.fit(X.reshape(-1, 1, 4), y, epochs=100, batch_size=128)  # Ubah bentuk X agar sesuai dengan input model

    last_date = data.index[-1].date()
    prediction_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=num_days, freq='D')
    X_pred = np.array([scaled_data[feature][-1] for feature in selected_features])

    y_pred = []
    for _ in range(num_days):
        pred = model.predict(X_pred.reshape(1, 1, 4))
        y_pred.append(pred[0, :])  # Menambahkan nilai prediksi (array 1x4) ke dalam daftar
        X_pred = np.array([pred])  # Mengubah bentuk input untuk prediksi berikutnya

    y_pred = np.array(y_pred)
    y_pred = np.transpose(y_pred)  # Transpose array untuk mendapatkan bentuk (4, num_days)
    y_pred = np.array([scaler.inverse_transform(y_pred[i].reshape(-1, 1)).flatten() for i in range(len(y_pred))])  # Melakukan inverse transform untuk setiap fitur

    predicted_dates = pd.Series([d.date() for d in prediction_dates], name='Date')
    predicted_data = pd.DataFrame(y_pred.T, columns=selected_features)
    predicted_data.insert(0, 'Date', predicted_dates)
    
    # Menyimpan DataFrame ke dalam file CSV
    predicted_data.to_csv('predict.csv', index=False)
