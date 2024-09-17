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

# Loading data from the CSV file
data = pd.read_csv('data/glass_data.csv')

# Preparing the data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Creating the results and models directories
results_dir = 'results/0.2_model_training_results/'
models_dir = 'results/0.2_trained_models/'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Defining scalers and models
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

# Initializing K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []  # List to store performance metrics
start_time = time.time()  # Start timing

# Loop through each scaler
for scaler_name, scaler in scalers.items():
    X_scaled = X if scaler is None else scaler.fit_transform(X)

    # Loop through each model
    for model_name, model in models.items():
        mse_scores = []  # List to store Mean Squared Error for each fold
        r2_scores = []  # List to store R² scores for each fold
        
        # Perform K-Fold Cross Validation
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Predict on test data
            mse_scores.append(mean_squared_error(y_test, y_pred))  # Calculate Mean Squared Error
            r2_scores.append(r2_score(y_test, y_pred))  # Calculate R² score

        # Calculating the mean and standard deviation of the scores
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        std_mse_percent = (std_mse / mean_mse) * 100
        std_r2_percent = (std_r2 / mean_r2) * 100 if mean_r2 != 0 else 0
    
        results.append([scaler_name, model_name, mean_mse, std_mse, std_mse_percent, mean_r2, std_r2, std_r2_percent])
        
        # Save the trained model to a file
        model_filename = f'{models_dir}{scaler_name}_{model_name}.joblib'
        joblib.dump(model, model_filename)

end_time = time.time()  # End timing
print(f"Total Time: {end_time - start_time:.2f} seconds")

# Converting results to a DataFrame and saving them to a CSV file
df_results = pd.DataFrame(results, columns=['Scaler', 'Model', 'Mean MSE', 'Std MSE', 'Std MSE %', 'Mean R²', 'Std R²', 'Std R² %'])
df_results.to_csv(f'{results_dir}model_performance.csv', index=False)
print(df_results)
