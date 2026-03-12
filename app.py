#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/readmission_model.pkl")

st.title("Hospital Readmission Risk Predictor")

age = st.number_input("Age")
bmi = st.number_input("BMI")
length_of_stay = st.number_input("Length of Stay")
medication_count = st.number_input("Medication Count")
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])

if st.button("Predict"):

    diabetes = 1 if diabetes=="Yes" else 0
    hypertension = 1 if hypertension=="Yes" else 0

    input_data = np.array([[age, bmi, length_of_stay,
                            medication_count,
                            diabetes, hypertension]])

    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Readmission Probability: {probability*100:.2f}%")
    
    if probability < 0.3:
        st.success("Low Risk")
    elif probability < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")


# In[ ]:




