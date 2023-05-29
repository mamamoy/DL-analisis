import mysql.connector
from datetime import datetime

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