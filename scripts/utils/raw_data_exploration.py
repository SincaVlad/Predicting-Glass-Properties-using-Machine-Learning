from sqlalchemy import create_engine
import pandas as pd

# Conectarea la baza de date SQLite
engine = create_engine('sqlite:///Glass_DB - Copy.db')  # Ajustează acest string la locația bazei tale de date

# Interogăm baza de date pentru a extrage toate datele din tabelul specificat
query = "SELECT * FROM SciGK;"  # Înlocuiește 'SciGK' cu numele corect al tabelului tău, dacă este diferit
data = pd.read_sql_query(query, engine)

# Calculăm numărul de înregistrări non-zero pentru fiecare coloană
non_zero_counts = data.apply(lambda x: (x != 0) & x.notnull()).sum()

# Crearea unui DataFrame pentru a afișa numele coloanelor și numărul de înregistrări non-zero
count_df = pd.DataFrame({'Column': non_zero_counts.index, 'Non-Zero Count': non_zero_counts.values})

# Salvarea rezultatelor într-un fișier text cu tab-uri ca delimitatori
count_df.to_csv('non_zero_counts.txt', index=False, sep='\t')

# Afișarea rezultatelor
print(count_df)
