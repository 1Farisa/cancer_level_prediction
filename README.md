# Cancer Level Prediction

This project aims to predict the cancer level of patients based on various health and environmental factors using machine learning algorithms.

## Description

The project leverages machine learning models to predict the cancer level of patients, classified into three categories: **Low**, **Medium**, and **High**. The dataset includes various features like age, gender, air pollution level, alcohol use, dust allergy, and more. The main goal is to develop a model that can predict the cancer level based on these inputs.

### Key Features:
- **Cancer Level Prediction:** Users can enter patient details such as age, gender, health conditions, and environmental factors to predict the cancer level.
- **Data Preprocessing:** The dataset was cleaned by removing unnecessary columns and transforming categorical data into numerical values suitable for training.
- **Model Training:** The project uses **Logistic Regression** for classification, with the ability to predict whether the cancer level is low, medium, or high based on the input data.

## Data:
The data is sourced from a health dataset, which includes:
- Patient attributes such as age, gender, and lifestyle habits.
- Health metrics like air pollution exposure, alcohol consumption, obesity levels, and more.

## Model:
- **Logistic Regression**: Used to train a model on the dataset to predict cancer risk levels.
- **Data Split**: 80% of the data is used for training, while 20% is used for testing.

## Requirements:
To run this project, you need to install the following Python libraries:
- pandas
- numpy
- scikit-learn
- streamlit

### Streamlit UI:
- The Streamlit app allows users to input their data through a simple interface with input fields such as age, gender, air pollution, smoking level, and more.
- Upon pressing the "Predict Cancer Level" button, the model predicts the cancer level, displaying the result as **Low**, **Medium**, or **High**.

### Additional Work:
- **Jupyter Notebook Implementation:** Apart from the Streamlit application, a Jupyter notebook has been created for cancer level prediction. This notebook implements various classification algorithms, including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM), to compare their performance and determine the most effective model.
- **Comparison of Algorithms:** The notebook compares different classification algorithms to evaluate which performs best in predicting cancer levels.
