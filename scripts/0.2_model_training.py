import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
import matplotlib.pyplot as plt
import joblib
import time
import os

# Încărcarea datelor din fișierul CSV
data = pd.read_csv('data/glass_data.csv')

# Pregătirea datelor
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Definirea scalerelor și modelelor
scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'QuantileTransformer': QuantileTransformer(output_distribution='uniform')
}

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Inițializarea K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Pregătirea structurii pentru rezultate
results = []
start_time = time.time()

# Crearea directorului de rezultate
results_dir = 'results/0.2_model_training_results/'
os.makedirs(results_dir, exist_ok=True)

for scaler_name, scaler in scalers.items():
    X_scaled = X if scaler is None else scaler.fit_transform(X)
    for model_name, model in models.items():
        mse_scores = []
        r2_scores = []
        
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        # Calcularea mediei și a deviației standard
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        std_mse_percent = (std_mse / mean_mse) * 100
        std_r2_percent = (std_r2 / mean_r2) * 100 if mean_r2 != 0 else 0

        # Salvarea rezultatelor
        results.append([scaler_name, model_name, mean_mse, std_mse, std_mse_percent, mean_r2, std_r2, std_r2_percent])
        
        # Salvarea modelului
        model_filename = f'{results_dir}{scaler_name}_{model_name}.joblib'
        joblib.dump(model, model_filename)

end_time = time.time()
print(f"Total Time: {end_time - start_time:.2f} seconds")

# Convertirea rezultatelor într-un DataFrame și salvarea acestora într-un fișier CSV
df_results = pd.DataFrame(results, columns=['Scaler', 'Model', 'Mean MSE', 'Std MSE', 'Std MSE %', 'Mean R²', 'Std R²', 'Std R² %'])
df_results.to_csv(f'{results_dir}model_performance.csv', index=False)

# Opțional, afișarea rezultatelor în consolă
print(df_results)
