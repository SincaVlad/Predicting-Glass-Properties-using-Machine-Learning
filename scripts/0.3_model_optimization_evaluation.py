import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import os

# Loading data from the CSV file
data = pd.read_csv('data/glass_data.csv')

# Preparing the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the results directory
results_dir = 'results/0.3_model_optimization_results/'
os.makedirs(results_dir, exist_ok=True)

# Defining hyperparameters for Random Search
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=20)],
    'max_depth': [None] + [int(x) for x in np.linspace(10, 200, num=20)],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['sqrt', 'log2', None, 0.5, 0.75]
}

# Initializing Random Forest model
rf = RandomForestRegressor(random_state=42)

# Initializing Random Search with 100 hyperparameter combinations and 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Training the model
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

# Displaying the best parameters found by Random Search
print(f"Best parameters: {random_search.best_params_}")
print(f"Random Search took {end_time - start_time:.2f} seconds")

# Using the best parameters to train the final model
best_rf = random_search.best_estimator_

# Evaluating the model's performance on the test set
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.2f}")

# Saving the trained model for later use
model_path = os.path.join(results_dir, 'Optimized_RandomForest_Model.joblib')
joblib.dump(best_rf, model_path)

# Generating plots

# Prediction vs. Actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valori Reale')
plt.ylabel('Predicții')
plt.title(f'Graficul de eroare predicție vs. valori reale\nRMSE: {rmse:.2f}, R²: {r2:.2f}')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'prediction_vs_actual.png'))
plt.close()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_rf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Antrenament')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validare încrucișată')

plt.xlabel('Dimensiunea Setului de Antrenament')
plt.ylabel('Acuratețe')
plt.title('Curba de învățare')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'learning_curve.png'))
plt.close()

# Feature Importance plot
importances = best_rf.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Importanța Caracteristicilor')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Caracteristică')
plt.ylabel('Importanță')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
plt.close()

print("Graficele și modelul antrenat au fost salvate.")
