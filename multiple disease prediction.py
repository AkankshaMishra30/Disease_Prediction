import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

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

# CSV file to store predictions
csv_file = "prediction_logs.csv"
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["timestamp", "disease", "result"])
    df.to_csv(csv_file, index=False)

def log_prediction(disease, result):
    df = pd.read_csv(csv_file)
    df = pd.concat([df, pd.DataFrame.from_records([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": disease,
        "result": result
    }])], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Health Assistant Dashboard',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction", 'Visual Analytics', 'Health Tips'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'bar-chart', 'capsule'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction')

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

    if st.button('Run Diabetes Prediction'):
        try:
            user_input = np.array([[float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                       Insulin, BMI, DiabetesPedigreeFunction, Age]]])
            std_input = diabetes_scaler.transform(user_input)
            diab_prediction = diabetes_model.predict(std_input)
            result = 'Positive' if diab_prediction[0] == 1 else 'Negative'
            log_prediction("Diabetes", result)
            st.success(f'The person is {"diabetic" if diab_prediction[0] == 1 else "not diabetic"}')
        except ValueError:
            st.error("Please enter valid numeric values.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')

    inputs = {}
    fields = ['Age', 'Sex', 'Chest Pain types', 'Resting BP', 'Cholesterol',
              'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate',
              'Exercise Induced Angina', 'Oldpeak', 'Slope', 'CA', 'Thal']

    cols = st.columns(3)
    for i, field in enumerate(fields):
        with cols[i % 3]:
            inputs[field] = st.text_input(field)

    if st.button('Run Heart Disease Prediction'):
        try:
            input_values = [float(inputs[f]) for f in fields]
            user_input = np.array([input_values])
            std_input = heartdisease_scaler.transform(user_input)
            prediction = heartdisease_model.predict(std_input)
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            log_prediction("Heart Disease", result)
            st.success(f'The person {"has" if prediction[0] == 1 else "does not have"} heart disease')
        except ValueError:
            st.error("Please enter valid numeric values.")

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")

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

    if st.button("Run Parkinson's Prediction"):
        try:
            user_input = np.array([[float(x) for x in user_values]])
            std_input = parkinsons_scaler.transform(user_input)
            prediction = parkinsons_model.predict(std_input)
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            log_prediction("Parkinson's", result)
            st.success(f'The person {"has" if prediction[0] == 1 else "does not have"} Parkinson's disease')
        except ValueError:
            st.error("Please enter valid numeric values.")

# Visual Analytics
if selected == 'Visual Analytics':
    st.title("📊 Real-Time Health Analytics")
    df = pd.read_csv(csv_file)
    
    st.subheader("Disease Test Counts")
    col1, col2 = st.columns(2)

    with col1:
        bar_data = df.groupby(['disease', 'result']).size().unstack().fillna(0)
        st.bar_chart(bar_data)

    with col2:
        pie_data = df['disease'].value_counts()
        st.write("### Predictions per Disease")
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    st.subheader("Raw Prediction Log")
    st.dataframe(df.tail(10))

# Health Tips
if selected == 'Health Tips':
    st.title("💡 Health Tips & Lifestyle Suggestions")
    st.markdown("""
    ### Diabetes
    - Maintain a balanced diet with low sugar intake
    - Regular exercise
    - Monitor blood sugar levels

    ### Heart Disease
    - Reduce salt and saturated fat
    - Quit smoking and limit alcohol
    - Manage stress and cholesterol

    ### Parkinson's
    - Follow physical therapy and balanced nutrition
    - Take medications on schedule
    - Get plenty of rest and mental support
    """)
