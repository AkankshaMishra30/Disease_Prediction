import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
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

# Setup CSV to log predictions
csv_file = 'predictions_log.csv'
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Disease", "Prediction", "Details", "Timestamp"])
    df.to_csv(csv_file, index=False)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Function to log the prediction to CSV
def log_prediction(disease, prediction, details):
    df = pd.read_csv(csv_file)
    new_entry = pd.DataFrame([[disease, prediction, details, pd.Timestamp.now()]], columns=["Disease", "Prediction", "Details", "Timestamp"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Input fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Run Diabetes Prediction'):
        try:
            user_input = np.array([[float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                       Insulin, BMI, DiabetesPedigreeFunction, Age]]])
            std_input = diabetes_scaler.transform(user_input)
            diab_prediction = diabetes_model.predict(std_input)
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            # Log the prediction
            log_prediction("Diabetes", diab_diagnosis, f"Details: {Pregnancies}, {Glucose}, {BloodPressure}, {SkinThickness}, {Insulin}, {BMI}, {DiabetesPedigreeFunction}, {Age}")
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Input fields
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
            # Log the prediction
            log_prediction("Heart Disease", heart_diagnosis, str(inputs))
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Input fields for Parkinson's Disease
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
            # Log the prediction
            log_prediction("Parkinson's Disease", parkinsons_diagnosis, str(user_values))
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(parkinsons_diagnosis)

# Display the prediction logs (dynamic tracking)
st.title("Prediction Log")
df = pd.read_csv(csv_file)
st.dataframe(df)

# Visualizing predictions dynamically
st.title("Prediction Statistics")

# Load prediction data
df = pd.read_csv(csv_file)

# Count how many predictions are positive or negative for each disease
diabetes_stats = df[df['Disease'] == 'Diabetes']['Prediction'].value_counts()
heart_stats = df[df['Disease'] == 'Heart Disease']['Prediction'].value_counts()
parkinsons_stats = df[df['Disease'] == "Parkinson's Disease"]['Prediction'].value_counts()

# Display bar charts
st.subheader("Diabetes Prediction Stats")
st.bar_chart(diabetes_stats)

st.subheader("Heart Disease Prediction Stats")
st.bar_chart(heart_stats)

st.subheader("Parkinson's Disease Prediction Stats")
st.bar_chart(parkinsons_stats)
