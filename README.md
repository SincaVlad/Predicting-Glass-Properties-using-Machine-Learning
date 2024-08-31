# Predicting Glass Properties using Machine Learning

This project focuses on predicting the properties of glass using various machine learning techniques. The project utilizes a dataset containing glass compositions and their corresponding transition temperatures (Tg), aiming to develop predictive models through a series of Python scripts.

## Overview

In this project, I utilized a dataset of approximately 29,000 glass compositions, derived from the SciGlass database, to predict the glass transition temperature (Tg) using machine learning algorithms such as Random Forest, Gradient Boosting, and Support Vector Regression. The best-performing model was optimized and evaluated using various metrics, with results indicating significant potential for the application of machine learning in materials science.

## Project Structure

- **data/** : Contains the dataset used for training and testing the machine learning models.
  - `glass_data.csv`: The processed dataset used for training and evaluation.
  - `original_data_reference.md`: Reference to the original SciGlass database and instructions on how to access it.

- **scripts/** : Contains Python scripts used for data exploration, model training, and optimization.
  - `data_exploration.py`: Script for exploring the dataset and generating visualizations.
  - `model_training.py`: Script for training multiple machine learning models and evaluating them.
  - `model_optimization_evaluation.py`: Script for hyperparameter optimization of the Random Forest model.
  - `model_usage.py`: Script for using the trained Random Forest model to predict Tg for a new glass composition.
  - **utils/**: Contains utility scripts for data handling and visualization.

- **results/** : Stores the outputs of the models, including trained models, performance metrics, and visualizations.
  - `model_performance.csv`: CSV file with model performance results (MSE, R², etc.).
  - `learning_curves.png`: Learning curve plot.
  - `feature_importance.png`: Feature importance plot.

## How to Use

1. **Data Preprocessing**:
   - Run `data_exploration.py` to explore the dataset.

2. **Model Training**:
   - Run `model_training.py` to train the machine learning models.

3. **Model Optimization**:
   - Use `model_optimization_evaluation.py` to optimize model hyperparameters and generate visualizations.

4. **Model Usage**:
   - Run `model_usage.py` to use the model .

## Dataset

The dataset used in this project is derived from the SciGlass database. Due to licensing restrictions, the raw data is not included in this repository. Please refer to `data/original_data_reference.md` for more information on accessing the original data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The original dataset is provided by the SciGlass database.
- This project is part of my master's dissertation at POLITEHNICA Bucharest National University of Science and Technology.
