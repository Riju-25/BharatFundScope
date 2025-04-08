import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Model
model = pickle.load(open('funding_model.pkl', 'rb'))

# Load Encoders
le_city = pickle.load(open('le_city.pkl', 'rb'))
le_industry = pickle.load(open('le_industry.pkl', 'rb'))
le_subvertical = pickle.load(open('le_subvertical.pkl', 'rb'))
le_investors = pickle.load(open('le_investors.pkl', 'rb'))
le_funding = pickle.load(open('le_funding.pkl', 'rb'))

st.title("Startup Funding Prediction App")

st.markdown("### Enter Startup Details")

# Create Dropdowns with Labels
city = st.selectbox("City Location", list(le_city.classes_))
industry = st.selectbox("Industry Vertical", list(le_industry.classes_))
subvertical = st.selectbox("Sub Vertical", list(le_subvertical.classes_))
investors = st.selectbox("Investors Name", list(le_investors.classes_))
funding = st.selectbox("Funding Type", list(le_funding.classes_))

num_investors = st.slider("Number of Investors", 1, 3)

# Prediction
if st.button("Predict Funding Amount"):
    city_enc = le_city.transform([city])[0]
    industry_enc = le_industry.transform([industry])[0]
    subvertical_enc = le_subvertical.transform([subvertical])[0]
    investors_enc = le_investors.transform([investors])[0]
    funding_enc = le_funding.transform([funding])[0]

    features = np.array([[city_enc, industry_enc, subvertical_enc, investors_enc, funding_enc, num_investors]])

    prediction = model.predict(features)

    st.success(f"Predicted Funding Amount: ₹ {prediction[0]:,.2f}")

