import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Startup Funding Predictor", page_icon="💰")
st.title("BharatFundScope: Indian Startup Predictor")

# Top 10 categories used during training + 'Other'
city_options = [
    'Bangalore', 'Mumbai', 'New Delhi', 'Hyderabad', 'Pune',
    'Chennai', 'Noida', 'Gurgaon', 'Ahmedabad', 'Jaipur', 'Other'
]
industry_options = [
    'E-commerce', 'Technology', 'Consumer Internet', 'Online Marketplace', 'Finance',
    'HealthCare', 'Logistics', 'Education', 'Food & Beverage', 'Travel', 'Other'
]
subvertical_options = [
    'Online Classifieds', 'Online Marketplace', 'Mobile Wallet', 'Food Delivery Platform',
    'Online Pharmacy', 'Crowdfunding', 'Real Estate Platform', 'Travel Marketplace',
    'Fashion Marketplace', 'Job Portal', 'Other'
]

# Frequencies from training data (dummy counts or use actual from training set)
city_freqs = {
    'Bangalore': 700, 'Mumbai': 550, 'New Delhi': 500, 'Hyderabad': 400, 'Pune': 350,
    'Chennai': 300, 'Noida': 250, 'Gurgaon': 200, 'Ahmedabad': 150, 'Jaipur': 100, 'Other': 50
}
industry_freqs = {
    'E-commerce': 650, 'Technology': 600, 'Consumer Internet': 550, 'Online Marketplace': 500,
    'Finance': 450, 'HealthCare': 400, 'Logistics': 350, 'Education': 300,
    'Food & Beverage': 250, 'Travel': 200, 'Other': 100
}
subvertical_freqs = {
    'Online Classifieds': 300, 'Online Marketplace': 270, 'Mobile Wallet': 250,
    'Food Delivery Platform': 230, 'Online Pharmacy': 210, 'Crowdfunding': 190,
    'Real Estate Platform': 170, 'Travel Marketplace': 150, 'Fashion Marketplace': 130,
    'Job Portal': 110, 'Other': 90
}

# Inputs
city = st.selectbox("City Location", city_options)
industry = st.selectbox("Industry Vertical", industry_options)
subvertical = st.selectbox("SubVertical", subvertical_options)
num_investors = st.number_input("Number of Investors", min_value=1, max_value=20, value=1)
tier = st.selectbox("Estimated Tier of Startup", ['Low', 'Medium', 'High'])

# Prepare input dataframe
X_input = pd.DataFrame([{
    'City Location_freq': city_freqs[city],
    'Industry Vertical_freq': industry_freqs[industry],
    'SubVertical_freq': subvertical_freqs[subvertical],
    'Num_Investors': num_investors
}])

# Load model and transformer for selected tier
with open(f"{tier}_model.pkl", 'rb') as f_model:
    model = pickle.load(f_model)
with open(f"{tier}_pt.pkl", 'rb') as f_pt:
    pt = pickle.load(f_pt)

# Predict
if st.button("Predict Funding 💵"):
    y_pred_trans = model.predict(X_input)[0]
    y_pred = pt.inverse_transform([[y_pred_trans]])[0][0]
    st.success(f"💰 Predicted Funding: ${y_pred:,.2f}")
