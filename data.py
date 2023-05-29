import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import mysql.connector

# Koneksi ke database MySQL
def create_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="analisis_sistem"  # Ganti dengan nama database yang sesuai
    )
    return conn

# Membuat tabel untuk menyimpan data
def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    table_query = '''
    CREATE TABLE IF NOT EXISTS saham (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume FLOAT
    )
    '''

    cursor.execute(table_query)
    conn.commit()

# Memasukkan data ke dalam database
def insert_data(data):
    conn = create_connection()
    cursor = conn.cursor()

    insert_query = "INSERT INTO saham (date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s)"

    # Menggunakan iterasi untuk memasukkan setiap baris data
    for row in data.itertuples(index=False):
        # Mengubah format tanggal dari string menjadi objek datetime
        date = datetime.strptime(row[0], "%Y-%m-%d").date()
        values = (date, row[1], row[2], row[3], row[4], row[5])
        cursor.execute(insert_query, values)

    conn.commit()
    conn.close()

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
def main():
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

if __name__ == "__main__":
    main()
