import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Încărcarea setului de date din locația specificată
data_path = 'data/glass_data.csv'
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Eroare: Fișierul '{data_path}' nu a fost găsit.")
    raise

# Variabila pentru salvarea graficelor
save_plots = True  # Setează la False pentru a afișa graficele în loc să le salvezi

# Crearea directorului pentru salvarea rezultatelor, dacă nu există
results_dir = 'results/0.1_data_exploration_results/'
os.makedirs(results_dir, exist_ok=True)

# Setarea stilului grafic
plt.style.use('ggplot')

# Generarea histogramelor pentru fiecare componentă chimică
for column in df.columns[:-1]:  # Exclude Tg
    data = df[df[column] != 0][column]  # Excluderea valorilor de 0 din calcul
    mean = data.mean()

    plt.figure(figsize=(14, 6))
    sns.histplot(data, kde=True, color='#1f77b4', edgecolor='black')
    
    plt.axvline(mean, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Medie: {mean:.2f}')
    
    plt.title(f'Histogramă {column[3:]} (Medie: {mean:.2f})')
    plt.xlabel(f'% Masic {column[3:]}')
    plt.ylabel('Frecvență')
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()

    if save_plots:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(results_dir, f'{column[3:]}_Histograma_{timestamp}.png'), dpi=300)
        plt.close()
    else:
        plt.show()

# Grafic distribuție Tg
mean_tg = df['TG'].mean()

plt.figure(figsize=(14, 6))
sns.histplot(df['TG'], kde=True, color='#1f77b4', edgecolor='black')
plt.axvline(mean_tg, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Medie: {mean_tg:.2f}')

plt.title(f'Histograma temperaturii de tranziție a sticlei (Tg) (Medie: {mean_tg:.2f})')
plt.xlabel('Tg (°C)')
plt.ylabel('Frecvență')
plt.xlim(0, None)
plt.legend()
plt.tight_layout()

if save_plots:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(results_dir, f'TG_Histograma_Standard_{timestamp}.png'), dpi=300)
    plt.close()
else:
    plt.show()

# Graficul de tip bară pentru numărul de compoziții
component_counts = (df.iloc[:, :-1] != 0).sum().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=component_counts.index.str[3:], y=component_counts.values, palette='deep')

plt.xticks(rotation=90)
plt.title(f'Numărul de compoziții care conțin fiecare componentă (Total compoziții: {df.shape[0]})')
plt.xlabel('Componentă')
plt.ylabel('Număr de compoziții')
plt.tight_layout()

if save_plots:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(results_dir, f'component_counts_{timestamp}.png'), dpi=300)
    plt.close()
else:
    plt.show()
