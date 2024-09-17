import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Loading data from the CSV file
data = pd.read_csv('data/glass_data.csv')

# Creating the results and models directories
results_dir = 'results/0.1_data_exploration_results/'
os.makedirs(results_dir, exist_ok=True)

# Setting the plot style
plt.style.use('ggplot')

# Generating histograms for each chemical component
for column in df.columns[:-1]:  # Exclude Tg
    data = df[df[column] != 0][column]  # Excluding zero values from the calculation
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
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(results_dir, f'{column[3:]}_Histograma_{timestamp}.png'), dpi=300)
    plt.close()


# Tg distribution plot
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

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(os.path.join(results_dir, f'TG_Histograma_Standard_{timestamp}.png'), dpi=300)
plt.close()

# Bar plot for the number of compositions
component_counts = (df.iloc[:, :-1] != 0).sum().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=component_counts.index.str[3:], y=component_counts.values, palette='deep')

plt.xticks(rotation=90)
plt.title(f'Numărul de compoziții care conțin fiecare componentă (Total compoziții: {df.shape[0]})')
plt.xlabel('Componentă')
plt.ylabel('Număr de compoziții')
plt.tight_layout()

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(os.path.join(results_dir, f'component_counts_{timestamp}.png'), dpi=300)
plt.close()
