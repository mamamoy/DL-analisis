import streamlit as st
import pandas as pd
import numpy as np
from database import create_connection, create_table, insert_data
from compare import compare_models
import plotly.express as px
def home():
    # st.title("Aplikasi Saham")
    # Tambahkan kode untuk tampilan halaman utama
    # Menampilkan data dari database
    def fetch_data():
        conn = create_connection()
        query = "SELECT * FROM saham"  # Ganti dengan query sesuai kebutuhan
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def chart_data():
        conn = create_connection()
        query = "SELECT date, close FROM saham"  # Ganti dengan query sesuai kebutuhan
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    # Menampilkan line chart dari data close dan date
    def show_line_chart(datachart):
        # Mengatur indeks datachart menjadi kolom 'date'
        datachart.set_index('date', inplace=True)

        # Menampilkan line chart dengan sumbu x berisi tanggal dan sumbu y berisi harga penutupan
        st.line_chart(datachart['close'])


    # Main function untuk menampilkan form dan menyimpan data

    st.title("Form Upload Dataset")
    st.divider()

    # Buat form untuk mengunggah file CSV
    file = st.file_uploader("Unggah file CSV", type=["csv"])
    submit_button = st.button("Submit")

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
        # Menampilkan DataFrame
        st.dataframe(dataframe, width=800)
        st.divider()
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
        'Model': average_rmse.index,
        'RMSE': average_rmse,
        'MSE': average_mse,
        'MAE': average_mae,
        'R-square': average_r2
    })

    # Normalisasi nilai metrik evaluasi
    normalized_metrics = {}
    for metric in average_metrics_df.columns[1:]:
        if metric == 'R-square':
            normalized_metric = average_metrics_df[metric]  # Tidak perlu dinormalisasi, karena semakin besar nilainya semakin baik
        else:
            normalized_metric = 1 - ((average_metrics_df[metric] - np.min(average_metrics_df[metric])) / (np.max(average_metrics_df[metric]) - np.min(average_metrics_df[metric])))
        normalized_metrics[metric] = normalized_metric

    # Mengalikan dengan bobot
    weighted_metrics = {}
    for metric in normalized_metrics:
        weighted_metric = normalized_metrics[metric] * weights[metric]
        weighted_metrics[metric] = weighted_metric

    # Menghitung skor total
    total_scores = np.sum(list(weighted_metrics.values()), axis=0)



    # Membuat DataFrame untuk skor total
    average_metrics_df['Total Score'] = total_scores

    # Mengurutkan model berdasarkan skor total
    sorted_models = average_metrics_df.sort_values(by='Total Score', ascending=False)

    # Mengembalikan model terbaik (model dengan skor tertinggi)
    best_model = sorted_models['Model'].iloc[0]
    st.markdown("<h3>Model Terbaik adalah model {}</h3>".format(best_model), unsafe_allow_html=True)

    

    st.dataframe(sorted_models, width=800) 
    
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
                weights = {'RMSE': 1.0, 'MSE': 1.0, 'MAE': 1.0, 'R-square': 1.0}
                rank_models('compare.csv', weights)
                

        except FileNotFoundError:
            st.warning("Belum melakukan run compare model.")

        try:
            # Menampilkan hasil prediksi dari file predicted_data.csv
            predicted_data = pd.read_csv('predicted_data.csv')
            if predicted_data is not None:
                st.subheader('Aktual vs Prediksi')
                st.dataframe(predicted_data, width=800)

                st.divider()
                
                # Line chart untuk memvisualisasikan perbandingan aktual dan prediksi
                st.subheader('Comparison Chart')
                # Convert 'Date' column to datetime
                predicted_data['Date'] = pd.to_datetime(predicted_data['Date'])

                # Create plot using plotly
                fig = px.line(predicted_data, x='Date', y=['Actual', 'Predicted_RNN', 'Predicted_GRU', 'Predicted_LSTM'], title='Actual vs. Predicted Prices')
                fig.update_layout(xaxis_title='Date', yaxis_title='Price')
                # fig.add_hline(y=1, line_color="red")rr
                fig.update_xaxes(showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False, zeroline=False)

                # Display plot using Streamlit
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
        except FileNotFoundError:
            st.warning("Belum melakukan run compare model.")

          

# Main function untuk mengatur routing berdasarkan URL
def main():
    st.sidebar.header("Menu")
    menu = st.sidebar.selectbox("Pilih halaman", ["Home", "Tuning Models", "Compare Models"])
    
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

if __name__ == "__main__":
    main()
