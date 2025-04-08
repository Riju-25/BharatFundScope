import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model
model = pickle.load(open('funding_model.pkl', 'rb'))

# Load Encoders with Mapping
encoders = {}
for col in ['City  Location', 'Industry Vertical', 'SubVertical', 'Investors Name', 'InvestmentnType']:
    le = pickle.load(open(f'le_{col.replace(" ", "_")}.pkl', 'rb'))
    encoders[col] = {
        'encoder': le,
        'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
    }

st.title("Startup Funding Amount Prediction")

st.subheader("Enter Startup Details")

# User Inputs (Show Labels)
city = st.selectbox("City Location", encoders['City  Location']['mapping'].keys())
industry = st.selectbox("Industry Vertical", encoders['Industry Vertical']['mapping'].keys())
subvertical = st.selectbox("Sub Vertical", encoders['SubVertical']['mapping'].keys())
investors = st.selectbox("Investors Name", encoders['Investors Name']['mapping'].keys())
funding = st.selectbox("Investment Type", encoders['InvestmentnType']['mapping'].keys())
year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
num_investors = st.slider("Number of Investors", 1, 3, 1)

# Map Labels to Encoded values
input_data = pd.DataFrame({
    'City  Location': [encoders['City  Location']['mapping'][city]],
    'Industry Vertical': [encoders['Industry Vertical']['mapping'][industry]],
    'SubVertical': [encoders['SubVertical']['mapping'][subvertical]],
    'Investors Name': [encoders['Investors Name']['mapping'][investors]],
    'InvestmentnType': [encoders['InvestmentnType']['mapping'][funding]],
    'Year': [year],
    'Month': [month],
    'Number of Investors': [num_investors]
})

if st.button('Predict Funding Amount'):
    log_pred = model.predict(input_data)
    prediction = np.expm1(log_pred[0])
    st.success(f"Predicted Funding Amount (USD): {prediction:,.2f}")
