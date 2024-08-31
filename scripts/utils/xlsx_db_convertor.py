import os
import pandas as pd
import sqlite3

# Calea către directorul care conține fișierele Excel de procesat
excel_path = "Tabel XLSX/Select" # Shimba cu Tabel XLSX/Prop
# Calea și numele fișierului bazei de date SQLite unde vor fi încărcate datele
sqlite_db_path = 'Glass_DB - Select.db' # Schimba cu Glass_DB - Prop

# Înființarea unei conexiuni la baza de date SQLite, fiind creată dacă nu există
conn = sqlite3.connect(sqlite_db_path)

# Găsirea tuturor fișierelor Excel în directorul specificat
file_list = [os.path.join(excel_path, file) for file in os.listdir(excel_path) if file.endswith(('.xls', '.xlsx'))]

# Iterează prin fiecare fișier Excel găsit
for file_path in file_list:
    # Deschide fișierul Excel pentru a accesa datele
    xls = pd.ExcelFile(file_path)
    # Obține numele tuturor foilor din fișierul Excel
    sheets = xls.sheet_names

    # Procesează fiecare foaie din fișierul Excel
    for sheet in sheets:
        # Citirea datelor din foaia curentă într-un DataFrame pandas
        df = pd.read_excel(file_path, sheet_name=sheet)
        # Extrage numele fișierului și folosește-l ca nume pentru tabelul SQLite
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        # Încarcă datele din DataFrame în tabelul corespunzător din baza de date SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False, chunksize=500)  # Scrierea în loturi pentru eficiență
        # Afișează un mesaj de confirmare pentru fiecare foaie procesată
        print(f"Loaded {sheet} of {file_path} into SQLite table {table_name}.")

# După încărcarea tuturor datelor, închide conexiunea la baza de date
conn.close()

# Afișează un mesaj final pentru a indica finalizarea procesului de încărcare
print("Introducerea de date din fișierele Excel în baza de date a fost realizată cu succes!")
