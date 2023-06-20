import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM
from tensorflow.keras.optimizers import Adam
from data import fetch_data

def compare_models(rnn_neurons, rnn_activation, rnn_learning_rate, gru_neurons, gru_activation, gru_learning_rate, lstm_neurons, lstm_activation, lstm_learning_rate):
    # Memuat data harga saham close BMRI
    data = fetch_data()  # Ubah dengan nama file yang sesuai

    # Forward fill missing date values
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.set_index('Date')
    data = data.resample('D').ffill()

    close_prices = data['Close'].values.reshape(-1, 1)

    # Melakukan penskalaan fitur
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Perform k-fold cross-validation
    k = 10
    kf = KFold(n_splits=k)
    results = {'Model': [], 'RMSE': [], 'MSE': [], 'MAE': [], 'R-square': []}

    predicted_data = pd.DataFrame()

    for train_index, test_index in kf.split(scaled_prices):
        train_scaled = scaled_prices[train_index]
        test_scaled = scaled_prices[test_index]
        train_data, test_data = train_scaled[:-1].reshape(-1, 1), test_scaled[:-1].reshape(-1, 1)
        train_target, test_target = train_scaled[1:], test_scaled[1:]  # Update the target indices

        # Build SimpleRNN model
        model_rnn = Sequential()
        model_rnn.add(SimpleRNN(rnn_neurons, activation=rnn_activation, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model_rnn.add(SimpleRNN(rnn_neurons, activation=rnn_activation))
        model_rnn.add(Dense(1))
        optimizer = Adam(learning_rate=rnn_learning_rate)
        model_rnn.compile(loss='mean_squared_error', optimizer=optimizer)

        # Build GRU model
        model_gru = Sequential()
        model_gru.add(GRU(gru_neurons, activation=gru_activation, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model_gru.add(GRU(gru_neurons, activation=gru_activation))
        model_gru.add(Dense(1))
        optimizer = Adam(learning_rate=gru_learning_rate)
        model_gru.compile(loss='mean_squared_error', optimizer=optimizer)

        # Build LSTM model
        model_lstm = Sequential()
        model_lstm.add(LSTM(lstm_neurons, activation=lstm_activation, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model_lstm.add(LSTM(lstm_neurons, activation=lstm_activation))
        model_lstm.add(Dense(1))
        optimizer = Adam(learning_rate=lstm_learning_rate)
        model_lstm.compile(loss='mean_squared_error', optimizer=optimizer)

        # Train the models
        model_rnn.fit(train_data, train_target, epochs=100, batch_size=128, validation_data=(test_data, test_target))
        model_gru.fit(train_data, train_target, epochs=100, batch_size=128, validation_data=(test_data, test_target))
        model_lstm.fit(train_data, train_target, epochs=100, batch_size=128, validation_data=(test_data, test_target))

        # Make predictions
        y_pred_rnn = model_rnn.predict(test_data)
        y_pred_gru = model_gru.predict(test_data)
        y_pred_lstm = model_lstm.predict(test_data)

        # Calculate RMSE, MAPE, MSE, MAE, R-square
        rmse_rnn = np.sqrt(mean_squared_error(test_target, y_pred_rnn))
        rmse_gru = np.sqrt(mean_squared_error(test_target, y_pred_gru))
        rmse_lstm = np.sqrt(mean_squared_error(test_target, y_pred_lstm))

        mse_rnn = mean_squared_error(test_target, y_pred_rnn)
        mse_gru = mean_squared_error(test_target, y_pred_gru)
        mse_lstm = mean_squared_error(test_target, y_pred_lstm)

        mae_rnn = mean_absolute_error(test_target, y_pred_rnn)
        mae_gru = mean_absolute_error(test_target, y_pred_gru)
        mae_lstm = mean_absolute_error(test_target, y_pred_lstm)

        r2_rnn = r2_score(test_target, y_pred_rnn)
        r2_gru = r2_score(test_target, y_pred_gru)
        r2_lstm = r2_score(test_target, y_pred_lstm)

        # Add results to dictionary
        results['Model'].extend(['SimpleRNN', 'GRU', 'LSTM'])
        results['RMSE'].extend([rmse_rnn, rmse_gru, rmse_lstm])
        results['MSE'].extend([mse_rnn, mse_gru, mse_lstm])
        results['MAE'].extend([mae_rnn, mae_gru, mae_lstm])
        results['R-square'].extend([r2_rnn, r2_gru, r2_lstm])

        # Create a list to store the data
        fold_predicted_data = []

        for i in range(len(test_index) - 1):
            fold_data = {
                'Date': data.index[test_index][i + 1],
                'Actual': scaler.inverse_transform(test_target)[i][0],
                'Predicted_RNN': scaler.inverse_transform(y_pred_rnn)[i][0],
                'Predicted_GRU': scaler.inverse_transform(y_pred_gru)[i][0],
                'Predicted_LSTM': scaler.inverse_transform(y_pred_lstm)[i][0]
            }
            fold_predicted_data.append(fold_data)


        # Create DataFrame for predicted data
        predicted_data = pd.DataFrame(fold_predicted_data)

        

    # Create DataFrame from results
    df_results = pd.DataFrame(results)

    # Save results to CSV file
    df_results.to_csv('compare.csv', index=False)

    # Save predicted data to CSV file
    predicted_data.to_csv('predicted_data.csv', index=False)

