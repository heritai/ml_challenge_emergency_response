# Emergency Response Time Prediction

This repository presents a robust machine learning framework designed to predict critical emergency response times. By integrating diverse datasets, including emergency event logs, vehicle telemetry, and routing information from OSRM (Open Source Routing Machine), the framework aims to deliver highly accurate and actionable predictions.

## Overview

The core objective of this project is to develop machine learning models capable of accurately forecasting one of two critical emergency response time metrics:

*   `delta selection-departure`: The time elapsed from an emergency vehicle's selection until its actual departure from the station.
*   `delta departure-presentation`: The duration from an emergency vehicle's departure until its arrival at the incident location.

The codebase is structured into modular components covering data loading, feature engineering, model selection, training, and evaluation, ensuring maintainability, scalability, and ease of extension for future enhancements.

## File Structure

```bash
emergency_response_prediction/
├── data/
│ ├── x_train.csv
│ ├── x_test.csv
│ └── y_train_u9upqBE.csv
├── src/
│ ├── data_loader.py # Handles loading raw data from CSV files.
│ ├── feature_engineering.py # Generates and preprocesses features for modeling.
│ ├── model_selection.py # Manages data splitting for training and validation.
│ ├── model_training.py # Orchestrates model training and evaluation.
│ └── utils.py # Provides general utility functions (e.g., logging, metrics calculation).
├── config.py # Configuration file for project paths and parameters.
├── requirements.txt # Lists Python dependencies.
└── README.md # This file.
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

**Important Note:** Due to licensing restrictions, the original dataset is not included in this repository. Users must provide their own data files following the specified structure.

## Usage

Follow these steps to set up and run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd emergency_response_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Provide Data Files:**
    Place your `x_train.csv`, `x_test.csv`, and `y_train_u9upqBE.csv` files into the `data/` directory.

4.  **Run the training script:**
    ```bash
    python src/model_training.py
    ```

Upon execution, this script will perform the following actions:

*   Load and preprocess the raw data.
*   Split the training dataset into training and validation sets.
*   Train multiple machine learning models (Linear Regression, Random Forest, and a foundational PyTorch Neural Network).
*   Evaluate each model on the validation set.
*   Select the best-performing model based on Mean Absolute Error (MAE).
*   Evaluate the best-performing model on the unseen test dataset.
*   Output key evaluation metrics: MAE, RMSE, and R-squared.

## Configuration

The `config.py` file serves as a central hub for all project configuration parameters, including:

*   Data file paths.
*   Definitions for categorical, numerical, and temporal features.
*   The specific target variable chosen for prediction.
*   Random seed for ensuring reproducibility.
*   Test set size for the validation split.

Users can easily modify this file to tailor the project to specific data inputs and requirements, ensuring flexibility and adaptability.

## Feature Engineering

The `feature_engineering.py` module is responsible for all feature engineering processes. Currently, it includes:

*   Calculating Haversine distance between geographical coordinates for spatial features.
*   Extracting temporal features from datetime columns (e.g., hour, day of week, month, quarter).
*   Imputing missing values to ensure data completeness.
*   Applying one-hot encoding to categorical features for model compatibility.
*   Standardizing numerical features to optimize model performance.

This module is designed for easy customization and extension; users are encouraged to add new features or modify existing ones. Special emphasis should be placed on leveraging and incorporating features derived from OSRM data, as these can significantly enhance predictive accuracy.

## Model Training

The `model_training.py` module orchestrates the training and evaluation of various machine learning models. Presently, it supports:

*   Linear Regression
*   Random Forest Regressor
*   A foundational PyTorch Neural Network

This module is designed for extensibility, allowing users to seamlessly integrate additional models or modify existing architectures. It also includes robust functionalities for model evaluation and selection based on validation set performance.

## Evaluation Metrics

The models' performance is assessed using the following standard evaluation metrics:

*   **Mean Absolute Error (MAE)**
*   **Root Mean Squared Error (RMSE)**
*   **R-squared (R²)**

## Next Steps

To further enhance this project and explore its full capabilities, consider the following next steps:

*   **Customize Feature Engineering:** Adapt the `feature_engineering.py` module to generate more pertinent and domain-specific features. Prioritize leveraging OSRM data and developing insightful interaction features to capture complex relationships.
*   **Hyperparameter Tuning:** Implement advanced hyperparameter tuning techniques (e.g., `GridSearchCV`, `RandomizedSearchCV` from scikit-learn, or more sophisticated methods like Bayesian Optimization) to fine-tune model parameters for superior performance.
*   **Explore Advanced Models:** Experiment with more sophisticated machine learning models, such as Gradient Boosting Machines (e.g., XGBoost, LightGBM, CatBoost) or more intricate Deep Learning architectures, to potentially uncover higher predictive power.
*   **Experiment Tracking:** Integrate dedicated tools like MLflow, Weights & Biases, or TensorBoard to systematically track experiments, manage model versions, and compare different model iterations efficiently, fostering better reproducibility and insights.
*   **Model Deployment:** Develop robust strategies for deploying the trained model into a production environment, enabling real-time or near real-time predictions for operational use.
*   **Address Class Imbalance:** If applicable, investigate and implement techniques such as SMOTE, ADASYN, or class weighting to effectively handle potential class imbalance within the target variables, ensuring fair model training.
*   **Survival Analysis:** If the `delta departure-presentation` metric is conceptualized as event duration data, explore survival analysis techniques and models to gain deeper, time-to-event specific insights.
*   **Geospatial Visualizations:** Leverage libraries like `geopandas`, `folium`, or `mapbox` to create compelling and interactive geospatial visualizations of emergency events, vehicle routes, and prediction outcomes, enhancing interpretability.

## Contributing

Contributions are welcome! Please submit pull requests with bug fixes, new features, or improved documentation.