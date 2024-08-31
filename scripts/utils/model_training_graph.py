import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Încărcarea datelor din CSV
df_results = pd.read_csv('model_performance.csv')

# Înlocuirea valorilor NaN cu "No_scale"
df_results['Scaler'].fillna('No_scale', inplace=True)

# Specificarea ordinii dorite pentru modele și scaleri
model_order = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'Gradient Boosting']
scaler_order = ['No_scale', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer']

# Crearea unui pivot table pentru Mean MSE și reordonarea conform cerințelor
mean_mse_pivot = df_results.pivot(index='Scaler', columns='Model', values='Mean MSE')
mean_mse_pivot = mean_mse_pivot.loc[scaler_order, model_order]

# Grafic pentru Mean MSE
plt.figure(figsize=(12, 8))
sns.heatmap(mean_mse_pivot, annot=True, fmt=".2f", cmap='viridis')
plt.title('Media deviațieilor standard a erori (Mean MSE)')
plt.xlabel('Model')
plt.ylabel('Scaler')
plt.tight_layout()
plt.savefig('Mean_MSE_Heatmap.png', dpi=300)
plt.close()

# Crearea unui pivot table pentru Mean R² și reordonarea conform cerințelor
mean_r2_pivot = df_results.pivot(index='Scaler', columns='Model', values='Mean R²')
mean_r2_pivot = mean_r2_pivot.loc[scaler_order, model_order]

# Grafic pentru Mean R²
plt.figure(figsize=(12, 8))
sns.heatmap(mean_r2_pivot, annot=True, fmt=".2f", cmap='viridis')
plt.title('Media coeficientilor de determinare (Mearn R²)')
plt.xlabel('Model')
plt.ylabel('Scaler')
plt.tight_layout()
plt.savefig('Mean_R2_Heatmap.png', dpi=300)
plt.close()

print("Graficele au fost salvate.")
