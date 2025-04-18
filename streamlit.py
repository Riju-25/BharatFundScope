import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Indian Startup Funding Predictor")

# Input dropdowns (match 'top 5' used during training)
city_options = ['Bangalore', 'Mumbai', 'New Delhi', 'Gurgaon', 'Noida', 'Other']
industry_options = ['Technology', 'E-commerce', 'Health Care', 'Education', 'Finance', 'Other']
subvertical_options = ['Mobile Apps', 'Online Marketplace', 'Analytics', 'EdTech', 'FinTech', 'Other']

# User Inputs
city = st.selectbox("City Location", city_options)
industry = st.selectbox("Industry Vertical", industry_options)
subvertical = st.selectbox("SubVertical", subvertical_options)
num_investors = st.number_input("Number of Investors", min_value=1, max_value=20, value=1)

# Frequency values (same as used in training)
city_freqs = {'Bangalore': 400, 'Mumbai': 300, 'New Delhi': 250, 'Gurgaon': 150, 'Noida': 100, 'Other': 50}
industry_freqs = {'Technology': 350, 'E-commerce': 300, 'Health Care': 200, 'Education': 150, 'Finance': 120, 'Other': 60}
subvertical_freqs = {'Mobile Apps': 250, 'Online Marketplace': 200, 'Analytics': 150, 'EdTech': 120, 'FinTech': 100, 'Other': 70}

# Convert to encoded features
X_input = pd.DataFrame([{
    'City Location_freq': city_freqs[city],
    'Industry Vertical_freq': industry_freqs[industry],
    'SubVertical_freq': subvertical_freqs[subvertical],
    'Num_Investors': num_investors
}])

# Choose tier
tier = st.selectbox("Estimated Funding Tier", ['Low', 'Medium', 'High'])

# Load corresponding model and transformer
with open(f'{tier}_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'{tier}_pt.pkl', 'rb') as f:
    pt = pickle.load(f)

# Predict
if st.button("Predict Funding Amount 💰"):
    y_pred_trans = model.predict(X_input)[0]
    y_pred = pt.inverse_transform([[y_pred_trans]])[0][0]
    st.success(f"🎯 Predicted Funding: ₹{y_pred:,.2f}")
