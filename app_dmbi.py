import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & encoders
model = pickle.load(open('funding_model.pkl', 'rb'))
le_city = pickle.load(open('le_city.pkl', 'rb'))
le_industry = pickle.load(open('le_industry.pkl', 'rb'))
le_subvertical = pickle.load(open('le_subvertical.pkl', 'rb'))
le_investors = pickle.load(open('le_investors.pkl', 'rb'))
le_funding = pickle.load(open('le_funding.pkl', 'rb'))

st.title("Startup Funding Prediction App")

# Dropdown options (based on original data used for encoding)
city_options = le_city.classes_
industry_options = le_industry.classes_
subvertical_options = le_subvertical.classes_
investor_options = le_investors.classes_
funding_options = le_funding.classes_

# User Inputs
city = st.selectbox("City Location", city_options)
industry = st.selectbox("Industry Vertical", industry_options)
subvertical = st.selectbox("SubVertical", subvertical_options)
investors = st.selectbox("Investors Name", investor_options)
funding_type = st.selectbox("Funding Type", funding_options)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
month = st.selectbox("Month", list(range(1, 13)))
num_investors = len(investors.split(','))

# Prepare Input Data
input_data = pd.DataFrame({
    'City  Location': [le_city.transform([city])[0]],
    'Industry Vertical': [le_industry.transform([industry])[0]],
    'SubVertical': [le_subvertical.transform([subvertical])[0]],
    'Investors Name': [le_investors.transform([investors])[0]],
    'InvestmentnType': [le_funding.transform([funding_type])[0]],
    'Year': [year],
    'Month': [month],
    'Number of Investors': [num_investors]
})

if st.button("Predict Funding Amount"):
    log_prediction = model.predict(input_data)
    prediction = np.expm1(log_prediction[0])
    st.success(f"Predicted Funding Amount: ${prediction:,.2f}")
