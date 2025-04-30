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

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []
if "prediction_count" not in st.session_state:
    st.session_state["prediction_count"] = 0

def track_prediction(disease, result):
    st.session_state.prediction_count += 1
    prediction = {
        "Prediction Number": st.session_state.prediction_count,
        "Disease": disease,
        "Prediction Result": result
    }
    st.session_state["predictions"].append(prediction)

# Tabs Navigation
st.title("\U0001F4D6 Multiple Disease Prediction System")
tabs = st.tabs(["📈 Visual Analytics", "👤 User Profile", "🌿 Health Tips", "🌊 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"])

# 1. Visual Analytics
with tabs[0]:
    st.subheader("Visual Analytics")
    st.markdown("View analytics of input trends and prediction distributions.")

    st.subheader("Prediction History")
    if len(st.session_state["predictions"]) > 0:
        history_df = pd.DataFrame(st.session_state["predictions"])
        st.dataframe(history_df)
    else:
        st.info("No predictions made yet.")

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
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# 2. User Profile
with tabs[1]:
    st.subheader("User Profile")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.write(f"Welcome, **{name}**! Your age is **{age}** and gender is **{gender}**.")
    st.info("Your recent predictions will appear in the Visual Analytics tab.")

# 3. Health Tips
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
    for tip in tips[disease]:
        st.markdown(f"- {tip}")

# 4. Diabetes Prediction
with tabs[3]:
    st.subheader('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age of the Person')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        BMI = st.text_input('BMI value')

    if st.button('Run Diabetes Prediction'):
        try:
            input_data = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness,
                                             Insulin, BMI, DiabetesPedigreeFunction, Age]]
            user_input = np.array([input_data])
            std_input = diabetes_scaler.transform(user_input)
            prediction = diabetes_model.predict(std_input)[0]
            result = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
            st.success(result)
            track_prediction("Diabetes", result)
        except ValueError:
            st.error("Please enter valid numeric values.")

# 5. Heart Disease Prediction
with tabs[4]:
    st.subheader('Heart Disease Prediction using ML')
    fields = ['Age', 'Sex', 'Chest Pain types', 'Resting BP', 'Cholesterol',
              'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate',
              'Exercise Induced Angina', 'Oldpeak', 'Slope', 'CA', 'Thal']
    
    inputs = {}
    cols = st.columns(3)
    for i, field in enumerate(fields):
        with cols[i % 3]:
            inputs[field] = st.text_input(field)

    if st.button('Heart Disease Test Result'):
        try:
            values = [float(inputs[field]) for field in fields]
            user_input = np.array([values])
            std_input = heartdisease_scaler.transform(user_input)
            prediction = heartdisease_model.predict(std_input)[0]
            result = 'The person has heart disease' if prediction == 1 else 'The person does not have heart disease'
            st.success(result)
            track_prediction("Heart Disease", result)
        except ValueError:
            st.error("Please enter valid numeric values.")

# 6. Parkinson's Prediction
with tabs[5]:
    st.subheader("Parkinson's Disease Prediction using ML")

    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
                'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    parkinsons_input = []
    for i in range(0, len(features), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(features):
                val = cols[j].text_input(features[i + j])
                parkinsons_input.append(val)

    if st.button("Parkinson's Test Result"):
        try:
            input_data = [float(x) for x in parkinsons_input]
            user_input = np.array([input_data])
            std_input = parkinsons_scaler.transform(user_input)
            prediction = parkinsons_model.predict(std_input)[0]
            result = "The person has Parkinson's disease" if prediction == 1 else "The person does not have Parkinson's disease"
            st.success(result)
            track_prediction("Parkinson's", result)
        except ValueError:
            st.error("Please enter valid numeric values.")
