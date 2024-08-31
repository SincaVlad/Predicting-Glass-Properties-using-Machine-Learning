import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Încărcarea datelor din fișierul CSV
data = pd.read_csv('Glass Data.csv')

# Pregătirea datelor: selectăm toate coloanele în afară de ultima pentru X (caracteristici) și ultima coloană pentru y (variabila de răspuns)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Împărțirea datelor în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Încărcarea modelului antrenat dintr-un fișier .joblib
best_rf = joblib.load('Optimized_RandomForest_Model.joblib')

# Generarea predicțiilor pe setul de testare
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Generarea graficelor

# Graficul de eroare predicție vs. valori reale
# Acest grafic compară valorile reale ale Tg (temperatura de tranziție a sticlei) cu valorile prezise de model.
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valori Reale')
plt.ylabel('Predicții')
plt.title(f'Grafic predicție vs. real \nRMSE: {rmse:.2f}, R²: {r2:.2f}')
plt.grid(True)
plt.savefig('prediction_vs_actual.png', dpi=300)
plt.close()

# Grafic de importanță a caracteristicilor (Feature Importance)
# Acest grafic arată importanța fiecărei caracteristici (compus chimic) în modelul de Random Forest.
importances = best_rf.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='deep')
plt.title('Importanța Caracteristicilor')
plt.xlabel('Importanță')
plt.ylabel('Caracteristică')
plt.grid(True)
plt.savefig('feature_importance.png', dpi=300)
plt.close()

print("Graficele au fost generate și salvate.")
