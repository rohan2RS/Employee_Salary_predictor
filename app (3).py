import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'linear_regression_model.pkl' is in the same directory.")
    st.stop()


dummy_data = pd.DataFrame({
    'Age': [30, 40, 50],
    'Gender': ['Male', 'Female', 'Male'],
    'Education_Level': ["Bachelor's", "Master's", "PhD"],
    'experience_years': [5, 10, 15],
    'Job_Title': ['Software Engineer', 'Data Analyst', 'Senior Manager'],
    'Salary': [70000, 100000, 150000]
})


gender_encoder = LabelEncoder()
dummy_data['Gender_encoded'] = gender_encoder.fit_transform(dummy_data['Gender'])

education_encoder = LabelEncoder()
dummy_data['Education_Level_encoded'] = education_encoder.fit_transform(dummy_data['Education_Level'])

job_title_encoder = LabelEncoder()
dummy_data['Job_Title_encoded'] = job_title_encoder.fit_transform(dummy_data['Job_Title'])


feature_scaler = StandardScaler()
feature_scaler.fit(dummy_data[['Age', 'experience_years']])

salary_scaler = StandardScaler()
salary_scaler.fit(dummy_data[['Salary']])


# Streamlit App
st.title("Employee Salary Prediction")
st.write("Enter employee details to predict their salary.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("Gender", ['Male', 'Female'])
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
years_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
job_title = st.text_input("Job Title", "Software Engineer") # Using text input for simplicity, ideally use selectbox with all possible job titles


# Prediction button
if st.button("Predict Salary"):
    # Convert categorical inputs to numerical using the encoders
    try:
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education_level])[0]
        
        job_title_encoded = job_title_encoder.transform([job_title])[0]

    except ValueError as e:
        st.error(f"Error encoding categorical features: {e}. Please ensure the input matches the training data categories.")
        st.stop()


    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Education_Level': [education_encoded],
        'experience_years': [years_experience],
        'Job_Title': [job_title_encoded]
    })

    # Scale the numerical features
    input_data[['Age', 'experience_years']] = feature_scaler.transform(input_data[['Age', 'experience_years']])

    # Make prediction
    predicted_salary_scaled = model.predict(input_data[['Age', 'Gender', 'Education_Level', 'experience_years', 'Job_Title']])

    # Inverse transform the scaled salary
    # Create a dummy array for inverse transformation
    dummy_array = np.zeros((predicted_salary_scaled.shape[0], 1)) # Assuming salary is the only feature for this scaler
    dummy_array[:, 0] = predicted_salary_scaled

    predicted_salary_original_scale = salary_scaler.inverse_transform(dummy_array)[:, 0]

    # Display the prediction
    st.subheader("Predicted Salary")
    st.write(f"${predicted_salary_original_scale[0]:,.2f}")
