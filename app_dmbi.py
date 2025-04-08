import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open('funding_model.pkl', 'rb'))

# Load LabelEncoders
le_city = pickle.load(open('le_city.pkl', 'rb'))
le_industry = pickle.load(open('le_industry.pkl', 'rb'))
le_subvertical = pickle.load(open('le_subvertical.pkl', 'rb'))
le_investors = pickle.load(open('le_investors.pkl', 'rb'))
le_funding = pickle.load(open('le_funding.pkl', 'rb'))

st.title("Startup Funding Prediction App")

st.markdown("### Enter Startup Details")

# Dropdowns for categorical features
city = st.selectbox("City Location", le_city.classes_)
industry = st.selectbox("Industry Vertical", le_industry.classes_)
subvertical = st.selectbox("Sub Vertical", le_subvertical.classes_)
investors = st.selectbox("Investors Name", le_investors.classes_)
funding = st.selectbox("Funding Type", le_funding.classes_)

# Slider for Number of Investors
num_investors = st.slider("Number of Investors", 1, 3, step=1)

if st.button("Predict Funding Amount"):
    # Encoding categorical values
    city_encoded = le_city.transform([city])[0]
    industry_encoded = le_industry.transform([industry])[0]
    subvertical_encoded = le_subvertical.transform([subvertical])[0]
    investors_encoded = le_investors.transform([investors])[0]
    funding_encoded = le_funding.transform([funding])[0]

    features = np.array([[city_encoded, industry_encoded, subvertical_encoded,
                          investors_encoded, funding_encoded, num_investors]])

    prediction = model.predict(features)

    st.success(f"Predicted Funding Amount: ₹ {prediction[0]:,.2f}")

