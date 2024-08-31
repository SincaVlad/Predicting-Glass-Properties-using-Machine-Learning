import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Încărcarea setului de date
df = pd.read_csv('Glass Data.csv')

# Variabila pentru salvarea graficelor
save_plots = True

# Setarea stilului grafic
plt.style.use('ggplot')

# Generarea histogramelor pentru fiecare componentă chimică
for column in df.columns[:-1]:  # Exclude Tg
    data = df[df[column] != 0][column]
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
        plt.savefig(f'{column[3:]}_Histograma.png', dpi=300)
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
    plt.savefig('TG_Histograma_Standard.png', dpi=300)
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
    plt.savefig('component_counts.png', dpi=300)
    plt.close()
else:
    plt.show()
