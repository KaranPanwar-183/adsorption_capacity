import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title='Biochar Capacity Predictor with SHAP', layout='wide')

st.title('🌿 Biochar Adsorption Capacity Predictor')
st.markdown('Predict the adsorption capacity and see the SHAP feature contributions.')

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_capacity_model.joblib')

model = load_model()

# Sidebar inputs
st.sidebar.header('Input Parameters')
def user_input_features():
    nc_ratio = st.sidebar.slider('N/C Ratio', 0.0, 0.2, 0.05)
    initial_conc = st.sidebar.slider('Initial Concentration (mg/L)', 0.5, 100.0, 10.0)
    surface_area = st.sidebar.number_input('Surface Area (m²/g)', value=500.0)
    nitrogen = st.sidebar.slider('Nitrogen content (%)', 0.0, 5.0, 1.5)
    pollutant = st.sidebar.selectbox('Pollutant', ['IBF', 'CBZ', 'ALA', 'DIU', 'SIM', 'CAR', 'PYR', 'TEB', 'ATE', 'EE2', 'NXP', 'DCF', 'IBU', 'NPX'])
    
    return {'N/C': nc_ratio, 'Initial concentration': initial_conc, 'Surface area': surface_area, 'N': nitrogen, 'Pollutant': pollutant}

input_dict = user_input_features()

if st.button('Predict & Analyze'):
    # In a real app, you must reconstruct the exact feature columns (e.g., 58 columns)
    # For this demo, we create a placeholder dataframe matching the model's expectation
    try:
        # Assuming model was trained on X_train.columns
        # We simulate the input row here
        feature_names = model.feature_names_in_
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # Map inputs to the dataframe
        for key, val in input_dict.items():
            if key in input_df.columns:
                input_df[key] = val
        
        # Handle one-hot encoding for the selected pollutant
        pollutant_col = f'Pollutant_{input_dict["Pollutant"]}'
        if pollutant_col in input_df.columns:
            input_df[pollutant_col] = 1.0
            
        # Set other likely constants used in training
        if 'Adsorption type_Single' in input_df.columns: input_df['Adsorption type_Single'] = 1.0
        if 'Wastewater type_Synthetic' in input_df.columns: input_df['Wastewater type_Synthetic'] = 1.0

        # Prediction
        prediction = model.predict(input_df)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Capacity", f"{prediction:.2f} mg/g")
            
        with col2:
            st.subheader("SHAP Local Explanation")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            fig, ax = plt.subplots()
            shap.plots.bar(shap.Explanation(shap_values[0], 
                                            base_values=explainer.expected_value, 
                                            data=input_df.iloc[0], 
                                            feature_names=feature_names), max_display=10, show=False)
            st.pyplot(plt.gcf())
            
    except Exception as e:
        st.error(f"Error in prediction/SHAP calculation: {e}")
        st.info("Ensure your model features match the input structure.")
