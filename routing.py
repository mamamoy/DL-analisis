import streamlit as st
import pandas as pd
import numpy as np
from database import create_connection, create_table, insert_data
def home():
    st.title("Aplikasi Saham")
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

    st.title("Aplikasi Pengunggahan Data Saham")

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
                st.dataframe(lstm_df)

        except FileNotFoundError:
            st.warning("Belum melakukan tuning model.")
          

# Main function untuk mengatur routing berdasarkan URL
def main():
    st.sidebar.header("Menu")
    menu = st.sidebar.selectbox("Pilih halaman", ["Home", "Tuning Model"])
    
    if menu == "Home":
        home()
    elif menu == "Tuning Model":
        rnn_tuning()
        st.divider()
        gru_tuning()
        st.divider()
        lstm_tuning()

if __name__ == "__main__":
    main()
