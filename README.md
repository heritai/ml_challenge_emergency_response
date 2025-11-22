# Emergency Response Time Prediction

This repository provides a machine learning framework for predicting emergency response times. It integrates data on emergency events, vehicle locations, and routing information from OSRM (Open Source Routing Machine) to deliver accurate predictions.

## Overview

The primary objective of this project is to develop a machine learning model capable of accurately predicting one of two critical emergency response time metrics:

*   `delta selection-departure`: The duration between an emergency vehicle's selection and its actual departure.
*   `delta departure-presentation`: The time elapsed from an emergency vehicle's departure to its arrival at the event location.

The codebase is organized into modular components for data loading, feature engineering, model selection, training, and evaluation, ensuring maintainability, scalability, and ease of extension.

## File Structure

```bash
emergency_response_prediction/
├── data/
│ ├── x_train.csv
│ ├── x_test.csv
│ └── y_train_u9upqBE.csv
├── src/
│ ├── data_loader.py # Loads data from CSV files.
│ ├── feature_engineering.py # Creates and preprocesses features.
│ ├── model_selection.py # Splits data into training and validation sets.
│ ├── model_training.py # Trains and evaluates machine learning models.
│ ├── utils.py # Utility functions (e.g., logging).
├── config.py # Configuration file for paths and parameters.
├── requirements.txt # List of Python dependencies.
├── README.md # This file.
```

## Dependencies

This project requires the following Python libraries:

*   pandas
*   numpy
*   scikit-learn
*   geopy
*   torch

Install them using pip:

```bash
pip install -r requirements.txt
```

### Data Requirements

The `data/` directory expects the following CSV files:

*   `x_train.csv`: Training features.
*   `x_test.csv`: Testing features.
*   `y_train_u9upqBE.csv`: Training target variables.

## Usage

Follow these steps to set up and run the project:

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd emergency_response_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Provide data files:**
    Place `x_train.csv`, `x_test.csv`, and `y_train_u9upqBE.csv` into the `data/` directory.
    **Note:** Due to copyright, the original dataset is not included in this repository.

4.  **Run the training script:**
    ```bash
    python src/model_training.py
    ```

This script will perform the following actions:

*   Load and preprocess the raw data.
*   Split the training dataset into training and validation sets.
*   Train multiple machine learning models (Linear Regression, Random Forest, and a PyTorch Neural Network).
*   Evaluate each model on the validation set.
*   Select the best-performing model based on Mean Absolute Error (MAE).
*   Evaluate the selected best model on the unseen test data.
*   Print key evaluation metrics: MAE, RMSE, and R-squared.

## Configuration

The `config.py` file centralizes all project configuration parameters, including:

*   Data file paths.
*   Definitions for categorical, numerical, and temporal features.
*   The target variable to be predicted.
*   Random seed for ensuring reproducibility.
*   Test set size for the validation split.

Users can modify this file to customize the project to specific data inputs and requirements.

## Feature Engineering

The `feature_engineering.py` module handles all feature engineering processes, which currently include:

*   Calculating the Haversine distance between geographical coordinates.
*   Extracting temporal features from datetime columns (e.g., hour, day of week, month).
*   Imputing missing values.
*   Applying one-hot encoding to categorical features.
*   Standardizing numerical features.

This module is designed for easy customization; feel free to add new features or modify existing ones. Special consideration should be given to incorporating features derived from OSRM data.

## Model Training

The `model_training.py` module is responsible for training and evaluating various machine learning models. Currently, it supports:

*   Linear Regression
*   Random Forest Regressor
*   A basic PyTorch Neural Network

This module is extensible, allowing users to integrate additional models or modify the existing architectures. It also encompasses functionalities for model evaluation and selection based on validation set performance.

## Evaluation Metrics

The performance of the models is assessed using the following standard evaluation metrics:

*   **Mean Absolute Error (MAE)**
*   **Root Mean Squared Error (RMSE)**
*   **R-squared**

## Next Steps

To further enhance this project and explore its capabilities, consider the following:

*   **Customize Feature Engineering:** Adapt the `feature_engineering.py` module to generate more pertinent features for your specific dataset. Prioritize leveraging OSRM data and developing interaction features.
*   **Hyperparameter Tuning:** Implement advanced tuning techniques such as `GridSearchCV` or `RandomizedSearchCV` from scikit-learn to optimize model hyperparameters for superior performance.
*   **Explore Advanced Models:** Experiment with more sophisticated machine learning models, including Gradient Boosting Machines (e.g., XGBoost, LightGBM, CatBoost) or more complex Deep Learning architectures.
*   **Experiment Tracking:** Integrate tools like MLflow or TensorBoard to systematically track experiments, manage model versions, and compare different model iterations efficiently.
*   **Model Deployment:** Develop strategies for deploying the trained model into a production environment for real-time predictions.
*   **Address Class Imbalance:** If applicable, investigate techniques such as SMOTE or class weighting to handle potential class imbalance in the target variables.
*   **Survival Analysis:** If the `delta departure-presentation` metric is treated as event data, explore survival analysis techniques and models to gain deeper insights into event durations.
*   **Geospatial Visualizations:** Utilize libraries like `geopandas` and `folium` to create interactive geospatial visualizations of emergency events and vehicle routes.

## Contributing

Contributions are welcome! Please submit pull requests with bug fixes, new features, or improved documentation.