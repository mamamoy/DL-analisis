import streamlit as st
import pandas as pd
import numpy as np
from database import create_connection, create_table, insert_data, delete_data, fetch_data
from compare import compare_models
from predict import predict_prices
import plotly.express as px
# from data import fetch_data
from datetime import datetime, timedelta
import plotly.graph_objects as go
def home():
    # Menampilkan data dari database
    def fetch_data():
        try:
            conn = create_connection()
            query = "SELECT * FROM saham"  # Ganti dengan query sesuai kebutuhan
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Periksa apakah dataframe kosong
            if df.empty:
                raise FileNotFoundError("Database kosong, silahkan memasukkan dataset!")

            return df

        except FileNotFoundError as e:
            st.warning(str(e))

    def chart_data():
        try:
            conn = create_connection()
            query = "SELECT date, close FROM saham"  # Ganti dengan query sesuai kebutuhan

            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            conn.close()

            if not result:
                raise FileNotFoundError("Data tidak ditemukan pada tabel saham!")

            df = pd.DataFrame(result, columns=['date', 'close'])
            return df
        except FileNotFoundError as e:
            st.warning(str(e))
        except Exception as e:
            st.warning("Terjadi kesalahan dalam menjalankan query SQL: " + str(e))

    
    # Menampilkan line chart dari data close dan date
    def show_line_chart(datachart):
        # Mengatur indeks datachart menjadi kolom 'date'
        datachart.set_index('date', inplace=True)

        # Menampilkan line chart dengan sumbu x berisi tanggal dan sumbu y berisi harga penutupan
        st.line_chart(datachart['close'])


    # Main function untuk menampilkan form dan menyimpan data

    st.title("Sistem Informasi Prediksi Harga Saham")
    st.divider()

    st.subheader("Input Dataset")
    col1, col2 = st.columns([7,2])
    with st.container():
        with col1:
            # Buat form untuk mengunggah file CSV
            file = st.file_uploader("Unggah file CSV", type=["csv"])
        with col2:
            submit_button = st.button("Submit")
            delete_button = st.button("Delete Dataset") 

    if delete_button:
        delete_data()

    if file is not None and submit_button:
        # Baca file CSV menjadi DataFrame
        df = pd.read_csv(file)

        # Membuat tabel di database jika belum ada
        create_table()

        # Memasukkan data ke dalam database
        insert_data(df)

        st.success("Data berhasil disimpan ke database.")
    
    dataframe = fetch_data()

    

    datachart = chart_data()
    if dataframe is not None:
        st.divider()
        st.subheader("Tabel Data Saham")
        # Menampilkan DataFrame
        st.dataframe(dataframe, width=800)
        st.divider()
        st.subheader("Chart Data Saham")
        # Menampilkan line chart
        show_line_chart(datachart)



def rnn_tuning():
    st.title("Hyperparameter Tuning Models")
    st.divider()
    col1, col2 = st.columns(2)

    with st.container():
        with col1:
            # Tambahkan kode untuk tampilan halaman prediksi
            st.write("Hyperparameter RNN Model Tuning")
        
        with col2:
            if st.button('Run RNN tuning'):
                def run_RNN_tuning():
                    import subprocess
                    subprocess.run(['python', 'RNN_tuning.py'], check=True)
                run_RNN_tuning()
        try:
            rnn_df = pd.read_csv('RNN_tuning_result.csv')
            if rnn_df is not None:

                # Dapatkan parameter terbaik berdasarkan indeks
                best_row = rnn_df['rank_test_score'] == 1
                best_params = rnn_df.loc[best_row, 'params'].values[0]
                best_rmse = rnn_df.loc[best_row, 'RMSE'].values[0]
                st.write("Kombinasi parameter terbaik: ", best_params)
                st.write("Nilai RMSE: ", best_rmse)
                st.dataframe(rnn_df)
        except FileNotFoundError:
            st.warning("Belum melakukan tuning model.")

def gru_tuning():
    col1, col2 = st.columns(2)

    with st.container():
        with col1:
            # Tambahkan kode untuk tampilan halaman prediksi
            st.write("Hyperparameter GRU Model Tuning")
        
        with col2:
            if st.button('Run GRU tuning'):
                def run_GRU_tuning():
                    import subprocess
                    subprocess.run(['python', 'GRU_tuning.py'], check=True)
                run_GRU_tuning()
        try:
            gru_df = pd.read_csv('GRU_tuning_result.csv')
            if gru_df is not None:

                # Dapatkan parameter terbaik berdasarkan indeks
                best_row = gru_df['rank_test_score'] == 1
                best_params = gru_df.loc[best_row, 'params'].values[0]
                best_rmse = np.sqrt(-gru_df.loc[best_row, 'mean_test_score'].values[0])
                st.write("Kombinasi parameter terbaik: ",best_params)
                st.write("Nilai RMSE: ", best_rmse)
                st.dataframe(gru_df)

        except FileNotFoundError:
            st.warning("Belum melakukan tuning model.")

def lstm_tuning():
    col1, col2 = st.columns(2)

    with st.container():
        with col1:
            # Tambahkan kode untuk tampilan halaman prediksi
            st.write("Hyperparameter LSTM Model Tuning")
        
        with col2:
            if st.button('Run LSTM tuning'):
                def run_LSTM_tuning():
                    import subprocess
                    subprocess.run(['python', 'LSTM_tuning.py'], check=True)
                run_LSTM_tuning()
        try:
            lstm_df = pd.read_csv('LSTM_tuning_result.csv')
            if lstm_df is not None:

                # Dapatkan parameter terbaik berdasarkan indeks
                best_row = lstm_df['rank_test_score'] == 1
                best_params = lstm_df.loc[best_row, 'params'].values[0]
                best_rmse = np.sqrt(-lstm_df.loc[best_row, 'mean_test_score'].values[0])
                st.write("Kombinasi parameter terbaik: ",best_params)
                st.write("Nilai RMSE: ", best_rmse)
                st.dataframe(lstm_df)

        except FileNotFoundError:
            st.warning("Belum melakukan tuning model.")

def rank_models(compare_file, weights):
    # Memuat file compare.csv ke dalam DataFrame
    compare_df = pd.read_csv(compare_file)

    # Menghitung rata-rata metrik evaluasi
    average_rmse = compare_df.groupby("Model")["RMSE"].mean()
    average_mse = compare_df.groupby("Model")["MSE"].mean()
    average_mae = compare_df.groupby("Model")["MAE"].mean()
    average_r2 = compare_df.groupby("Model")["R-square"].mean()

    # Menggabungkan rata-rata metrik evaluasi menjadi satu DataFrame
    average_metrics_df = pd.DataFrame({
        # 'Model': average_rmse.index,
        'RMSE': average_rmse,
        'MSE': average_mse,
        'MAE': average_mae,
        'R-square': average_r2
    })

    st.dataframe(average_metrics_df, width=800) 
    
    st.divider()

def run_compare_models():
    with st.container():
        # Menampilkan input form untuk hyperparameter
        st.subheader('RNN Parameters')
        rnn_neurons = st.number_input('RNN Neurons', min_value=0)
        rnn_activation = st.selectbox('RNN Activation', ['sigmoid', 'tanh', 'relu'])
        rnn_learning_rate = st.text_input('RNN Learning Rate', value='0.001')

        st.divider()

        st.subheader('GRU Parameters')
        gru_neurons = st.number_input('GRU Neurons', min_value=0)
        gru_activation = st.selectbox('GRU Activation', ['sigmoid', 'tanh', 'relu'])
        gru_learning_rate = st.text_input('GRU Learning Rate', value='0.001')

        st.divider()
            
        st.subheader('LSTM Parameters')
        lstm_neurons = st.number_input('LSTM Neurons', min_value=0)
        lstm_activation = st.selectbox('LSTM Activation', ['sigmoid', 'tanh', 'relu'])
        lstm_learning_rate = st.text_input('LSTM Learning Rate', value='0.001')

        st.divider()

        
        if st.button('Run Compare Models'):
            # Menjalankan fungsi compare_models
            compare_models(rnn_neurons, rnn_activation, float(rnn_learning_rate), gru_neurons, gru_activation, float(gru_learning_rate), lstm_neurons, lstm_activation, float(lstm_learning_rate))

        try:
            st.divider()
            st.subheader("Hasil Evaluasi dengan K = 10")
            # Menampilkan hasil dari file compare.csv
            compare_df = pd.read_csv('compare.csv')
            if compare_df is not None:
                st.dataframe(compare_df, width=800)
                st.divider()
                st.subheader("Nilai Rata-rata Hasil Evaluasi")
                weights = {'RMSE': 1.0, 'MSE': 1.0, 'MAE': 1.0, 'R-square': 1.0}
                rank_models('compare.csv', weights)
            # Menampilkan hasil prediksi dari file predicted_data.csv
            predicted_data = pd.read_csv('predicted_data.csv')
            if predicted_data is not None:
                st.subheader('Aktual vs Prediksi')
                st.dataframe(predicted_data, width=800)

                st.divider()
                
                # Line chart untuk memvisualisasikan perbandingan aktual dan prediksi
                st.subheader('Chart Perbandingan Model')
                # Convert 'Date' column to datetime
                predicted_data['Date'] = pd.to_datetime(predicted_data['Date'])

                # Create plot using plotly
                fig = px.line(predicted_data, x='Date', y=['Actual', 'Predicted_RNN', 'Predicted_GRU', 'Predicted_LSTM'], title='Aktual vs. Prediksi')
                fig.update_layout(xaxis_title='Date', yaxis_title='Price')
                # fig.add_hline(y=1, line_color="red")rr
                fig.update_xaxes(showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False, zeroline=False)

                # Display plot using Streamlit
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
                

        except FileNotFoundError:
            st.warning("Belum melakukan proses compare model.")

def run_predict():
    st.title("Prediksi Harga Saham")
    st.divider()

    model_option = st.selectbox('Pilih Model', ('RNN', 'GRU', 'LSTM'))
    col1, col2, col3 = st.columns([4, 4, 2])

    with st.container():

        with col1:
            data = fetch_data()
            if data is not None:
                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
                # Mendapatkan tanggal terakhir dari data
                last_date = data['Date'].max().date() if not data.empty else pd.Timestamp.now().date()
                # Tambahkan kode untuk tampilan halaman prediksi
                start_date = st.date_input("Mulai dari", last_date, disabled=True)

        with col2:
            end_date = st.date_input("Sampai dengan", datetime.now())
            # Validasi tanggal minimum
        minimum_date = datetime(2023, 2, 28).date()
        if end_date < minimum_date:
            st.warning("Tanggal yang dipilih harus setidaknya 28 Februari 2023.")

        num_days = (end_date - start_date).days

        with col3:
            if st.button('Run prediksi'):
                predict_prices(num_days=num_days, model_name=model_option)

        st.write("Jumlah hari yang akan diprediksi", num_days, "hari")

        try:
            actual_data = fetch_data()
            data = pd.read_csv('predict.csv')
            if data is not None:
                st.divider()
                st.subheader('Tabel hasil prediksi')
                st.dataframe(data, width=800)
                st.divider()
                st.subheader("Chart hasil prediksi")
                # Membuat trace untuk data aktual
                trace_actual = go.Scatter(x=actual_data['Date'], y=actual_data['Close'], name='Aktual',
                                         line=dict(color='blue'))

                # Membuat trace untuk data prediksi
                trace_predicted = go.Scatter(x=data['Date'], y=data['Close'], name='Prediksi',
                                            line=dict(color='red'))

                # Menggabungkan trace menjadi satu
                data = [trace_actual, trace_predicted]

                # Mengatur layout chart
                layout = go.Layout(title='Prediksi Harga')

                # Membuat figure dan menambahkan data dan layout
                fig = go.Figure(data=data, layout=layout)

                # Menampilkan chart menggunakan plotly_chart
                st.plotly_chart(fig)
        except FileNotFoundError:
            st.warning("Belum melakukan prediksi.")


          

# Main function untuk mengatur routing berdasarkan URL
def main():
    st.sidebar.header("Menu")
    menu = st.sidebar.selectbox("Pilih halaman", ["Home", "Tuning Models", "Compare Models", "Predict Price"])
    
    if menu == "Home":
        home()
    elif menu == "Tuning Models":
        rnn_tuning()
        st.divider()
        gru_tuning()
        st.divider()
        lstm_tuning()
    elif menu == "Compare Models":
        run_compare_models()
    elif menu == "Predict Price":
        run_predict()

if __name__ == "__main__":
    main()
