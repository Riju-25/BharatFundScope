import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Startup Funding Prediction", page_icon="🚀", layout="centered")

st.title("🚀 Startup Funding Prediction App")

# Load Pickle Files
def load_pickle(file_name):
    return pickle.load(open(os.path.join(os.path.dirname(__file__), file_name), 'rb'))

model = load_pickle('funding_model.pkl')
le_city = load_pickle('le_city.pkl')
le_industry = load_pickle('le_industry.pkl')
le_subvertical = load_pickle('le_subvertical.pkl')
le_investors = load_pickle('le_investors.pkl')
le_funding = load_pickle('le_funding.pkl')

st.markdown("## Enter Startup Details")

city = st.selectbox("City Location", le_city.classes_)
industry = st.selectbox("Industry Vertical", le_industry.classes_)
subvertical = st.selectbox("Sub Vertical", le_subvertical.classes_)
investors = st.selectbox("Investors", le_investors.classes_)
funding_type = st.selectbox("Funding Type", le_funding.classes_)
num_investors = st.slider("Number of Investors", 1, 3, 1)

if st.button("Predict Funding Amount"):
    input_data = np.array([
        le_city.transform([city])[0],
        le_industry.transform([industry])[0],
        le_subvertical.transform([subvertical])[0],
        le_investors.transform([investors])[0],
        le_funding.transform([funding_type])[0],
        num_investors
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Funding Amount: ₹ {np.round(prediction, 2)}")

st.markdown("---")
st.markdown("Developed with ❤️ by *Your Name*")
