# Customer Churn Prediction

This project predicts whether a customer will churn (leave the service) from a telecommunications company using machine learning. The model is trained on a dataset that contains various features such as customer demographics, account details, and services they use.

## Overview

In this project, a **Random Forest Classifier** is used to predict churn. The data is processed, encoded, scaled, and balanced using **SMOTE** before being used to train the model. A Flask API is then used to serve the model, allowing real-time predictions based on user input.


## Dataset

The dataset used in this project is the **Telco Customer Churn dataset**, which includes various attributes related to customer account information, demographics, and whether they churned.

You can find the dataset in the `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` file.

## Model Training

### Model

- **Model Type**: Random Forest Classifier
- **Accuracy**: The trained model achieved an accuracy of **85.75%** on the test data. This means that the model is able to correctly predict whether a customer will churn with an accuracy of **85.75%**. This high accuracy suggests the model performs reliably in predicting customer churn.

### Preprocessing

1. **Missing Values**: Missing values in the `TotalCharges` column were filled with the median of the column.
2. **Encoding**: Categorical features were encoded using **Label Encoding** to convert them into numerical values.
3. **Scaling**: Numerical features were scaled using **StandardScaler** to normalize the data and improve model performance.
4. **Balancing**: The dataset was imbalanced, with significantly more non-churned customers than churned ones. To address this, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to balance the dataset by generating synthetic samples for the minority class (churned customers).

### Model Evaluation

The model was evaluated using several metrics including:
- **Accuracy**: 85.75%
- **Classification Report**: Precision, recall, and F1-score for each class (churned vs non-churned).

To train the model, open the `model_training.ipynb` notebook and execute the cells.

## Flask API

Once the model is trained, a Flask API is used to serve the model and allow users to make predictions by sending HTTP requests.

### Dependencies
The project requires the following libraries:

Flask: For building the API.
pandas: For data manipulation and analysis.
numpy: For numerical operations.
scikit-learn: For machine learning and model training.
imbalanced-learn: For SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
gunicorn: WSGI server for running the Flask app in production.

```bash
pip install -r requirements.txt
