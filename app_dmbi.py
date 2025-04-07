import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model and Encoders
model = pickle.load(open('funding_model.pkl', 'rb'))
le_city = pickle.load(open('city_encoder.pkl', 'rb'))
le_industry = pickle.load(open('industry_encoder.pkl', 'rb'))

# Load Dataset (only to extract unique city & industry for dropdown)
data = pd.read_excel('startup_funding.xlsx')

unique_cities = sorted(data['City  Location'].dropna().unique())
unique_industries = sorted(data['Industry Vertical'].dropna().unique())

# Streamlit App UI
st.title("🚀 Startup Funding Prediction App")
st.subheader("Predict Funding Amount based on City and Industry Vertical")

# Dropdown Menus
city = st.selectbox("Select City Location", unique_cities)
industry = st.selectbox("Select Industry Vertical", unique_industries)

# Prediction Button
if st.button("Predict Funding Amount"):
    # Transform input using encoders
    city_code = le_city.transform([city])[0]
    industry_code = le_industry.transform([industry])[0]

    X_new = np.array([[city_code, industry_code]])

    prediction = model.predict(X_new)

    st.success(f"Estimated Funding Amount : ₹ {prediction[0]:,.2f}")
