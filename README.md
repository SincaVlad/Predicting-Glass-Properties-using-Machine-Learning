# Predicting Glass Properties using Machine Learning

This project focuses on predicting the properties of glass using various machine learning techniques. The goal is to develop predictive models for the glass transition temperature (Tg) based on a dataset of glass compositions. This is achieved through a series of Python scripts organized in a structured directory.

## Overview

The project utilizes a dataset of approximately 29,000 glass compositions from the SciGlass database to predict the glass transition temperature (Tg) using machine learning algorithms such as Random Forest, Gradient Boosting, and Support Vector Regression. The best-performing model was optimized and evaluated using various metrics, demonstrating the potential of machine learning in materials science.

## Project Structure

- **scripts/**: Contains Python scripts for data handling, model training, and evaluation.
  - **data/**: 
    - `glass_data.csv`: Processed dataset for training and evaluation.
    - `original_data_reference.md`: Reference to the SciGlass database and instructions for accessing it.
  - **results/**: 
    - `0.1_data_exploration_results/`: Results from data exploration.
    - `0.2_model_training_results/`: Contains `model_performance.csv` with model performance metrics.
    - `0.2_trained_models/`: Trained models.
    - `0.3_model_optimization_results/`: Contains `feature_importance.png`, `learning_curve.png`, and `prediction_vs_actual.png`.
    - `0.3_optimized_model/`: Contains the optimized Random Forest model `Optimized_RandomForest_Model.joblib`.
  - **utils/**: Utility scripts for various tasks.
    - `model_evaluation_graph.py`: Script for generating model evaluation graphs.
    - `model_training_graph.py`: Script for visualizing model training results.
    - `raw_data_exploration.py`: Script for exploring raw data.
    - `xlsx_db_convertor.py`: Script for converting Excel databases.
  - `0.1_data_exploration.py`: Script for exploring the dataset and generating visualizations.
  - `0.2_model_training.py`: Script for training machine learning models.
  - `0.3_model_optimization_evaluation.py`: Script for optimizing and evaluating model hyperparameters.
  - `0.4_model_usage.py`: Script for predicting Tg with the trained model.

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **LICENSE**: License details for the project.
- **README.md**: This file.

## How to Use

1. **Data Preprocessing**:
   - Run `0.1_data_exploration.py` to explore and visualize the dataset.

2. **Model Training**:
   - Execute `0.2_model_training.py` to train the machine learning models.

3. **Model Optimization**:
   - Use `0.3_model_optimization_evaluation.py` to optimize model hyperparameters and visualize the results.

4. **Model Usage**:
   - Run `0.4_model_usage.py` to make predictions with the optimized model.

## Dataset

The dataset used in this project is sourced from the SciGlass database. Due to licensing restrictions, the raw data is not included in this repository. Please refer to `scripts/data/original_data_reference.md` for details on accessing the original dataset.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The original dataset is provided by the SciGlass database.
- This project is part of my master's dissertation at POLITEHNICA Bucharest National University of Science and Technology.

