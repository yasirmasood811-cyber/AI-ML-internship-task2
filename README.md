# AI-ML-internship-task2
# Task 2:Customer Churn Prediction using Scikit-learn Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The pipeline performs data preprocessing, model training, hyperparameter tuning, evaluation, and model export in a reusable and production-ready workflow.

## Objectives

* Predict whether a customer will churn.
* Automate preprocessing using Scikit-learn Pipeline.
* Compare Logistic Regression and Random Forest models.
* Optimize model performance using GridSearchCV.
* Save the trained pipeline using Joblib for future predictions.

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer demographic information, account details, service subscriptions, and whether the customer has churned.

**Target Variable:**

* Churn (Yes/No)

## Features

* Data cleaning and preprocessing
* Handling missing values
* Numerical feature scaling using StandardScaler
* Categorical feature encoding using OneHotEncoder
* ColumnTransformer for mixed data preprocessing
* Logistic Regression model
* Random Forest model
* Hyperparameter tuning with GridSearchCV
* Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* Exporting the complete trained pipeline using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Running the Project

Run the Python script:

```bash
python churn_pipeline.py
```

The script will:

1. Load the dataset.
2. Preprocess the data.
3. Train Logistic Regression and Random Forest models.
4. Perform hyperparameter tuning using GridSearchCV.
5. Evaluate model performance.
6. Save the best pipeline as `customer_churn_pipeline.pkl`.

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Model Export

The trained model pipeline is saved using:

```python
joblib.dump(best_model, "customer_churn_pipeline.pkl")
```

To load the saved model:

```python
loaded_model = joblib.load("customer_churn_pipeline.pkl")
```

## Skills Demonstrated

* Machine Learning Pipeline Construction
* Data Preprocessing
* Feature Engineering
* Hyperparameter Tuning
* Model Evaluation
* Production-Ready Model Deployment
* Model Serialization using Joblib
  
# Task 3: Multimodal Machine Learning – Housing Price Prediction

## Objective

The objective of this project is to predict housing prices using both house images and structured tabular data. A Convolutional Neural Network (CNN) is used to extract visual features from house images, while a dense neural network processes numerical housing attributes. Both feature sets are combined to build a multimodal regression model.

---

## Dataset

This project uses:

- Custom Housing Dataset (housing_20.csv)
- Custom House Images (House_Images folder)

### Tabular Features

- Bedrooms
- Bathrooms
- Area
- Garage
- Age
- Price (Target Variable)
- ImageName

### Image Dataset

The `House_Images` folder contains 20 house images corresponding to the `ImageName` column in the CSV file.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Methodology

1. Load housing dataset.
2. Load and preprocess house images.
3. Normalize image pixel values.
4. Standardize tabular features.
5. Build a CNN for image feature extraction.
6. Build a Dense Neural Network for tabular features.
7. Combine image and tabular features using feature fusion.
8. Train the multimodal regression model.
9. Predict housing prices.
10. Evaluate model performance using MAE and RMSE.

---

## Model Architecture

### Image Branch

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- Flatten
- Dense

### Tabular Branch

- Dense
- Dense

### Feature Fusion

- Concatenate
- Dense
- Dropout
- Output Layer (Regression)

---

## Evaluation Metrics

The model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## Results

The notebook generates:

- Training & Validation Loss Graph
- Actual vs Predicted Prices Graph
- Prediction Error Plot
- MAE
- RMSE

---

## Skills Gained

- Multimodal Machine Learning
- Convolutional Neural Networks (CNNs)
- Image Feature Extraction
- Feature Fusion
- Regression Modeling
- Model Evaluation (MAE & RMSE)

---

# Task 1:News Topic Classifier Using BERT

## Overview

This project focuses on classifying news headlines into four categories: World, Sports, Business, and Sci/Tech using the BERT (bert-base-uncased) transformer model. Instead of training a model from scratch, a pre-trained BERT model is fine-tuned on the AG News dataset to improve classification performance. The project also includes a simple Gradio interface that allows users to enter a news headline and receive its predicted category in real time.

## Dataset

The project uses the AG News Dataset from the Hugging Face Datasets library. It contains 120,000 training samples and 7,600 testing samples across four news categories:

World
Sports
Business
Sci/Tech

## Technologies Used
Python
Hugging Face Transformers
Hugging Face Datasets
BERT (bert-base-uncased)
NumPy
Evaluate
Gradio
Workflow
Load the AG News dataset.
Tokenize the news headlines using the BERT tokenizer.
Fine-tune the pre-trained bert-base-uncased model.
Evaluate the model using Accuracy and Weighted F1-score.
Save the trained model and deploy it with Gradio for live predictions.

## Learning Outcomes
Natural Language Processing with Transformers
Transfer Learning using BERT
Text Classification
Model Evaluation using Accuracy and F1-score
Deployment of an NLP model with Gradio

## Conclusion

This project demonstrates how transfer learning can be applied to build an accurate news topic classifier using BERT. Fine-tuning a pre-trained transformer model significantly improves text classification performance while reducing training time. The Gradio interface provides an easy way to interact with the model and classify news headlines in real time.
## Author
Yasir Masood
