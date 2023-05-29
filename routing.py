import streamlit as st
import pandas as pd
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



def tuning():
    st.title("Tuning Model")
    # Tambahkan kode untuk tampilan halaman prediksi

# Main function untuk mengatur routing berdasarkan URL
def main():
    st.sidebar.header("Menu")
    menu = st.sidebar.selectbox("Pilih halaman", ["Home", "Tuning Model"])
    
    if menu == "Home":
        home()
    elif menu == "Tuning Model":
        tuning()

if __name__ == "__main__":
    main()
