
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title='Universal Biochar Capacity Predictor', layout='wide')

st.title('🌿 Full-Scale Biochar Adsorption Predictor')
st.markdown('Enter physicochemical properties for a high-precision adsorption capacity prediction based on the champion Random Forest model.')

@st.cache_resource
def load_model():
    return joblib.load('random_forest_capacity_model.joblib')

model = load_model()
feature_names = model.feature_names_in_

# Sidebar inputs with optimized generic ranges
st.sidebar.header('1. Pyrolysis & Physical Properties')
pyro_temp = st.sidebar.slider('Pyrolysis Temp (°C)', 200, 1200, 700)
pyro_time = st.sidebar.slider('Pyrolysis Time (min)', 0, 1440, 120)
sa = st.sidebar.number_input('Surface Area (m²/g)', 0.0, 5000.0, 450.0)
pore_vol = st.sidebar.number_input('Pore Volume (cm³/g)', 0.0, 3.0, 0.25)
pore_size = st.sidebar.slider('Avg Pore Size (nm)', 0.1, 100.0, 2.5)

st.sidebar.header('2. Elemental Composition (%)')
c_cont = st.sidebar.slider('Carbon (C) %', 0.1, 100.0, 80.0)
h_cont = st.sidebar.slider('Hydrogen (H) %', 0.0, 15.0, 1.5)
o_cont = st.sidebar.slider('Oxygen (O) %', 0.0, 60.0, 5.0)
n_cont = st.sidebar.slider('Nitrogen (N) %', 0.0, 30.0, 1.5)
ash = st.sidebar.slider('Ash Content %', 0.0, 90.0, 10.0)

st.sidebar.header('3. Adsorption Conditions')
pollutant = st.sidebar.selectbox('Pollutant', ['IBF', 'CBZ', 'ALA', 'DIU', 'SIM', 'CAR', 'PYR', 'TEB', 'ATE', 'EE2', 'NXP', 'DCF', 'IBU', 'NPX'])
adsorbent = st.sidebar.selectbox('Adsorbent Type', ['PB600', 'PB800', 'GCRB', 'GCRB-N', 'PSB', 'PSBOX-A', 'C-Biochar', 'PAC', 'Pristine SCG biochar', 'Alkali-modified SCG biochars', 'Pristine SCW Biochar', 'NaOH-activated SCW biochars', 'CB', 'MCB', 'AMCB'])
ww_type = st.sidebar.selectbox('Wastewater Type', ['Synthetic', 'Lake water', 'Ground water', 'Secondary effluent'])
ads_type = st.sidebar.selectbox('Adsorption Type', ['Single', 'Competative'])

initial_conc = st.sidebar.number_input('Initial Conc (mg/L)', 0.01, 2000.0, 10.0)
solution_ph = st.sidebar.slider('Solution pH', 0.0, 14.0, 7.0)
adsorp_time = st.sidebar.number_input('Adsorption Time (min)', 1, 10000, 180)
temp = st.sidebar.slider('Adsorption Temp (°C)', 0, 100, 25)
dosage = st.sidebar.number_input('Adsorbent Dosage (g/L)', 0.01, 10.0, 0.1)
rpm = st.sidebar.slider('Stirring Speed (RPM)', 0, 500, 150)
vol = st.sidebar.number_input('Solution Volume (L)', 0.001, 1.0, 0.05)
ion_conc = st.sidebar.number_input('Ion Concentration (mol/L)', 0.0, 1.0, 0.0)
humic = st.sidebar.number_input('Humic Acid (mg/L)', 0.0, 100.0, 0.0)

if st.button('Predict Adsorption Capacity'):
    # Initialize 58-column input row
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Direct Mappings
    input_df['Pyrolysis temperature '] = pyro_temp
    input_df['Pyrolysis time'] = pyro_time
    input_df['C'], input_df['H'], input_df['O'], input_df['N'] = c_cont, h_cont, o_cont, n_cont
    input_df['Ash'], input_df['Surface area'], input_df['Pore volume'] = ash, sa, pore_vol
    input_df['Average pore size'], input_df['Adsorption time'] = pore_size, adsorp_time
    input_df['Initial concentration'], input_df['Solution pH'] = initial_conc, solution_ph
    input_df['Adsorbent dosage'], input_df['Adsorption temperature'] = dosage, temp
    input_df['RPM'], input_df['Volume'] = rpm, vol
    input_df['Ion concentration'], input_df['Humic acid'] = ion_conc, humic

    # Recalculate Ratios to avoid feature mismatch
    input_df['H/C'] = h_cont / c_cont
    input_df['O/C'] = o_cont / c_cont
    input_df['N/C'] = n_cont / c_cont
    input_df['(O+N)/C'] = (o_cont + n_cont) / c_cont

    # Categorical One-Hot Logic
    if f'Pollutant_{pollutant}' in feature_names: input_df[f'Pollutant_{pollutant}'] = 1.0
    if f'Adsorbent_{adsorbent}' in feature_names: input_df[f'Adsorbent_{adsorbent}'] = 1.0
    if f'Wastewater type_{ww_type}' in feature_names: input_df[f'Wastewater type_{ww_type}'] = 1.0
    if f'Adsorption type_{ads_type}' in feature_names: input_df[f'Adsorption type_{ads_type}'] = 1.0

    # Prediction
    prediction = model.predict(input_df)[0]
    st.metric("Predicted Adsorption Capacity (qe)", f"{prediction:.4f} mg/g")

    # SHAP Explainer
    st.subheader("Feature Contribution (SHAP Analysis)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()
    shap.plots.bar(shap.Explanation(shap_values[0], base_values=explainer.expected_value, data=input_df.iloc[0], feature_names=feature_names), max_display=12, show=False)
    st.pyplot(plt.gcf())
