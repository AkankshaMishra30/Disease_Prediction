import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load models and scalers
model_path = "models"  

# Load Diabetes model and scaler
diabetes_model = pickle.load(open(os.path.join(model_path, "diabetes trained_model (1).sav"), "rb"))
diabetes_scaler = pickle.load(open(os.path.join(model_path, "diabetes_scaler.sav"), "rb"))

# Load Heart Disease model and scaler
heartdisease_model = pickle.load(open(os.path.join(model_path, "heartdisease_trained_model (1).sav"), "rb"))
heartdisease_scaler = pickle.load(open(os.path.join(model_path, "heartdisease_scaler.sav"), "rb"))

# Load Parkinson's model and scaler
parkinsons_model = pickle.load(open(os.path.join(model_path, "parkinsons_trained_model (1).sav"), "rb"))
parkinsons_scaler = pickle.load(open(os.path.join(model_path, "parkinsons_scaler.sav"), "rb"))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Run Diabetes Prediction'):
        try:
            user_input = np.array([[float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                       Insulin, BMI, DiabetesPedigreeFunction, Age]]])
            std_input = diabetes_scaler.transform(user_input)
            diab_prediction = diabetes_model.predict(std_input)
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    inputs = {}
    fields = ['Age', 'Sex', 'Chest Pain types', 'Resting BP', 'Cholesterol',
              'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate',
              'Exercise Induced Angina', 'Oldpeak', 'Slope', 'CA', 'Thal']

    cols = st.columns(3)
    for i, field in enumerate(fields):
        with cols[i % 3]:
            inputs[field] = st.text_input(field)

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            input_values = [float(inputs[f]) for f in fields]
            user_input = np.array([input_values])
            std_input = heartdisease_scaler.transform(user_input)
            heart_prediction = heartdisease_model.predict(std_input)
            heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
                'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    user_values = []
    for i in range(0, len(features), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(features):
                val = cols[j].text_input(features[i + j])
                user_values.append(val)

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            user_input = np.array([[float(x) for x in user_values]])
            std_input = parkinsons_scaler.transform(user_input)
            parkinsons_prediction = parkinsons_model.predict(std_input)
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(parkinsons_diagnosis)
