# Predicting Glass Transition Temperature Using Machine Learning

This repository contains the code and data used in my dissertation on predicting the glass transition temperature (Tg) of glass materials using various machine learning techniques. The project includes data preprocessing, model training, and optimization, as well as the evaluation of model performance.

## Overview

In this project, I utilized a dataset of approximately 29,000 glass compositions, derived from the SciGlass database, to predict the glass transition temperature (Tg) using machine learning algorithms such as Random Forest, Gradient Boosting, and Support Vector Regression. The best-performing model was optimized and evaluated using various metrics, with results indicating significant potential for the application of machine learning in materials science.

## Files and Directories

- **data/**
  - `glass_data.csv`: The processed dataset used for training and evaluation.
  - `original_data_reference.md`: Reference to the original SciGlass database and instructions on how to access it.

- **scripts/**
  - `data_preprocessing.py`: Python script for data preprocessing.
  - `model_training.py`: Script for training and evaluating machine learning models.
  - `model_optimization.py`: Script for optimizing model hyperparameters using Randomized Search.
  - `model_evaluation.py`: Script for final model evaluation and generating plots.
  - **utils/**: Contains utility scripts for data handling and visualization.

- **results/**
  - `model_performance.csv`: CSV file with model performance results (MSE, R², etc.).
  - `learning_curves.png`: Learning curve plot.
  - `feature_importance.png`: Feature importance plot.
  - **predictions/**: Contains prediction examples for new glass compositions.

## How to Use

1. **Data Preprocessing**:
   - Run `data_preprocessing.py` to prepare the dataset.

2. **Model Training**:
   - Run `model_training.py` to train the machine learning models.

3. **Model Optimization**:
   - Use `model_optimization.py` to optimize model hyperparameters.

4. **Model Evaluation**:
   - Run `model_evaluation.py` to evaluate the model and generate visualizations.

## Dataset

The dataset used in this project is derived from the SciGlass database. Due to licensing restrictions, the raw data is not included in this repository. Please refer to `data/original_data_reference.md` for more information on accessing the original data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The original dataset is provided by the SciGlass database.
- This project is part of my master's dissertation at [Your University Name].
