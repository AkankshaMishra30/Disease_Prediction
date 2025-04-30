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

diabetes_model = pickle.load(open(os.path.join(model_path, "diabetes trained_model (1).sav"), "rb"))
diabetes_scaler = pickle.load(open(os.path.join(model_path, "diabetes_scaler.sav"), "rb"))

heartdisease_model = pickle.load(open(os.path.join(model_path, "heartdisease_trained_model (1).sav"), "rb"))
heartdisease_scaler = pickle.load(open(os.path.join(model_path, "heartdisease_scaler.sav"), "rb"))

parkinsons_model = pickle.load(open(os.path.join(model_path, "parkinsons_trained_model (1).sav"), "rb"))
parkinsons_scaler = pickle.load(open(os.path.join(model_path, "parkinsons_scaler.sav"), "rb"))

# Setup CSV to log predictions
csv_file = 'predictions_log.csv'
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Disease", "Prediction", "Details", "Timestamp"])
    df.to_csv(csv_file, index=False)

# Tabs Navigation
st.title("\U0001F4D6 Multiple Disease Prediction System")
tabs = st.tabs(["📈 Visual Analytics", "👤 User Profile", "🌿 Health Tips", "🌊 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"])

# Visual Analytics
with tabs[0]:
    st.subheader("Visual Analytics")
    st.markdown("View analytics of input trends and prediction distributions.")

    dummy_data = pd.DataFrame({
        'Age': np.random.randint(20, 80, 50),
        'Glucose': np.random.randint(70, 180, 50),
        'BMI': np.round(np.random.uniform(18.5, 35.0, 50), 1),
        'Prediction': np.random.choice(['Diabetes', 'Heart', 'Parkinson\'s'], 50)
    })

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(dummy_data[['Age', 'Glucose']])
    with col2:
        st.line_chart(dummy_data[['BMI']])

    st.subheader("Disease Prediction Distribution")
    pred_counts = dummy_data['Prediction'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# User Profile
with tabs[1]:
    st.subheader("User Profile")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.write(f"Welcome, **{name}**! Your age is **{age}** and gender is **{gender}**.")
    st.info("Your recent predictions will appear here once a prediction is made.")

# Health Tips
with tabs[2]:
    st.subheader("Health Tips & Recommendations")
    disease = st.selectbox("Select a disease to view tips", ["Diabetes", "Heart Disease", "Parkinson's"])

    tips = {
        "Diabetes": [
            "Maintain a healthy diet (low sugar/carbs).",
            "Exercise regularly.",
            "Monitor blood sugar levels.",
            "Avoid smoking and excessive alcohol."
        ],
        "Heart Disease": [
            "Eat heart-healthy foods (low cholesterol/sodium).",
            "Manage stress and get regular sleep.",
            "Avoid tobacco and limit alcohol.",
            "Maintain a healthy weight."
        ],
        "Parkinson's": [
            "Engage in physical therapy and exercise.",
            "Eat a balanced diet rich in fiber.",
            "Consult doctors about medication.",
            "Join support groups for mental health."
        ]
    }

# Function to log the prediction to CSV
def log_prediction(disease, prediction, details):
    df = pd.read_csv(csv_file)
    new_entry = pd.DataFrame([[disease, prediction, details, pd.Timestamp.now()]], columns=["Disease", "Prediction", "Details", "Timestamp"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)
    
    for tip in tips[disease]:
        st.markdown(f"- {tip}")

# Diabetes Prediction Page
with tabs[3]:
    st.subheader('Diabetes Prediction using ML')

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
with tabs[4]:
    st.subheader('Heart Disease Prediction using ML')

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
with tabs[5]:
    st.subheader("Parkinson's Disease Prediction using ML")

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
