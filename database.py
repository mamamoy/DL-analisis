import mysql.connector
from datetime import datetime
import pandas as pd

# Koneksi ke database MySQL
def create_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="dataset_saham"  # Ganti dengan nama database yang sesuai
    )
    return conn

# Membuat tabel untuk menyimpan data
def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    table_query = '''
    CREATE TABLE IF NOT EXISTS saham (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Date DATE,
        Open FLOAT,
        High FLOAT,
        Low FLOAT,
        Close FLOAT,
        Volume FLOAT
    )
    '''

    cursor.execute(table_query)
    conn.commit()

# Memasukkan data ke dalam database
def insert_data(data):
    conn = create_connection()
    cursor = conn.cursor()

    # Memeriksa apakah tabel saham sudah ada
    cursor.execute("SHOW TABLES LIKE 'saham'")
    table_exists = cursor.fetchone()

    if not table_exists:
        create_table()

    insert_query = "INSERT INTO saham (Date, Open, High, Low, Close, Volume) VALUES (%s, %s, %s, %s, %s, %s)"

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

def delete_data():
    conn = create_connection()
    cursor = conn.cursor()

    truncate_query = '''
    TRUNCATE TABLE saham
    '''

    cursor.execute(truncate_query)
    conn.commit()
    conn.close()