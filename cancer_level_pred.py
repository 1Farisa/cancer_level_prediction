import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/cancer.csv")

# Drop the 'Patient Id' column as it is not necessary for prediction
df.drop('Patient Id', axis=1, inplace=True)

# Map the 'Level' column values to numerical values for model compatibility
scale_mapper = {'Low': 0, 'Medium': 1, 'High': 2}
df['Level'] = df['Level'].replace(scale_mapper)

# Define the feature matrix 'x' (input variables) and the target vector 'y' (the cancer level)
x = df.drop('Level', axis=1)  # Features: all columns except 'Level'
y = df['Level']  # Target: the 'Level' column

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(x_train, y_train)

# Streamlit UI
st.title("Cancer Level Prediction")  # Set the title of the app
st.write("Enter patient details below to predict the cancer level:")  # Prompt for input

# Create input fields for user input using Streamlit widgets
age = st.number_input('Age', min_value=0, max_value=120, value=30)  # Input for age
gender = st.selectbox('Gender', ['Male', 'Female'])  # Input for gender
gender = 0 if gender == 'Male' else 1  # Map gender to 0 (Male) or 1 (Female)
air_pollution = st.slider('Air Pollution Level', min_value=0, max_value=10, value=5)  # Air pollution level
alcohol_use = st.slider('Alcohol Use Level', min_value=0, max_value=10, value=5)  # Alcohol use level
dust_allergy = st.slider('Dust Allergy Level', min_value=0, max_value=10, value=5)  # Dust allergy level
occupational_hazards = st.slider('Occupational Hazards', min_value=0, max_value=10, value=5)  # Occupational hazards
genetic_risk = st.slider('Genetic Risk', min_value=0, max_value=10, value=5)  # Genetic risk level
chronic_lung_disease = st.slider('Chronic Lung Disease', min_value=0, max_value=10, value=5)  # Chronic lung disease
balanced_diet = st.slider('Balanced Diet Level', min_value=0, max_value=10, value=5)  # Balanced diet level
obesity = st.slider('Obesity Level', min_value=0, max_value=10, value=5)  # Obesity level
smoking = st.slider('Smoking Level', min_value=0, max_value=10, value=5)  # Smoking level
passive_smoker = st.slider('Passive Smoker Level', min_value=0, max_value=10, value=5)  # Passive smoker level
chest_pain = st.slider('Chest Pain Level', min_value=0, max_value=10, value=5)  # Chest pain level
coughing_of_blood = st.slider('Coughing of Blood Level', min_value=0, max_value=10, value=5)  # Coughing of blood
fatigue = st.slider('Fatigue Level', min_value=0, max_value=10, value=5)  # Fatigue level
weight_loss = st.slider('Weight Loss Level', min_value=0, max_value=10, value=5)  # Weight loss level
shortness_of_breath = st.slider('Shortness of Breath Level', min_value=0, max_value=10, value=5)  # Shortness of breath level
wheezing = st.slider('Wheezing Level', min_value=0, max_value=10, value=5)  # Wheezing level
swallowing_difficulty = st.slider('Swallowing Difficulty', min_value=0, max_value=10, value=5)  # Swallowing difficulty level
clubbing_of_finger_nails = st.slider('Clubbing of Finger Nails', min_value=0, max_value=10, value=5)  # Clubbing of finger nails
frequent_cold = st.slider('Frequent Cold Level', min_value=0, max_value=10, value=5)  # Frequent cold level
dry_cough = st.slider('Dry Cough Level', min_value=0, max_value=10, value=5)  # Dry cough level
snoring = st.slider('Snoring Level', min_value=0, max_value=10, value=5)  # Snoring level

# Create a DataFrame from the user input to pass to the model for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Air Pollution': [air_pollution],
    'Alcohol use': [alcohol_use],
    'Dust Allergy': [dust_allergy],
    'OccuPational Hazards': [occupational_hazards],
    'Genetic Risk': [genetic_risk],
    'chronic Lung Disease': [chronic_lung_disease],
    'Balanced Diet': [balanced_diet],
    'Obesity': [obesity],
    'Smoking': [smoking],
    'Passive Smoker': [passive_smoker],
    'Chest Pain': [chest_pain],
    'Coughing of Blood': [coughing_of_blood],
    'Fatigue': [fatigue],
    'Weight Loss': [weight_loss],
    'Shortness of Breath': [shortness_of_breath],
    'Wheezing': [wheezing],
    'Swallowing Difficulty': [swallowing_difficulty],
    'Clubbing of Finger Nails': [clubbing_of_finger_nails],
    'Frequent Cold': [frequent_cold],
    'Dry Cough': [dry_cough],
    'Snoring': [snoring]
})

# Button to trigger the prediction
if st.button('Predict Cancer Level'):
    prediction = model.predict(input_data)  # Make a prediction using the trained model
    # Map the numeric prediction back to the categorical label
    level_mapper = {0: 'Low', 1: 'Medium', 2: 'High'}
    st.write(f'The predicted cancer level is: **{level_mapper[prediction[0]]}**')  # Display the prediction
