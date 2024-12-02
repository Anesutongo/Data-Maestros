import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Ensure this filename matches your model's file

# Extract feature names from the model
try:
    feature_names = model.feature_names_in_
except AttributeError:
    # Fallback if feature_names_in_ is not available
    feature_names = [
        'Population', 'GDP PER CAPITA (USD)', 'GDP PER CAPITA PPP (USD)', 'Area (Km2)',
        'Transportation (Mt)', 'Other Fuel Combustion (Mt)', 'Manufacturing/Construction (Mt)',
        'Industrial Processes (Mt)', 'Fugitive Emissions (Mt)', 'Energy (Mt)',
        'Electricity/Heat (Mt)', 'Bunker Fuels (Mt)', 'Building (Mt)',
        'CO2 per Capita', 'Energy Intensity', 'Land-Use Change and Forestry (Mt)'
    ]

def predict_carbon_emission(input_features):
    # Ensure input DataFrame columns match the model's feature names
    input_data = pd.DataFrame([input_features], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title('Carbon Emission Prediction')
    st.write('Enter the details below to predict total carbon emissions (excluding LUCF):')

    # Input fields for the features in your dataset
    input_features = [
        st.number_input("Population", value=1000000),
        st.number_input("GDP Per Capita (USD)", value=2000.0),
        st.number_input("GDP Per Capita PPP (USD)", value=3000.0),
        st.number_input("Area (Km2)", value=500000),
        st.number_input("Transportation Emissions (Mt)", value=10.0),
        st.number_input("Other Fuel Combustion (Mt)", value=5.0),
        st.number_input("Manufacturing/Construction Emissions (Mt)", value=15.0),
        st.number_input("Industrial Processes Emissions (Mt)", value=8.0),
        st.number_input("Fugitive Emissions (Mt)", value=6.0),
        st.number_input("Energy Emissions (Mt)", value=20.0),
        st.number_input("Electricity/Heat Emissions (Mt)", value=12.0),
        st.number_input("Bunker Fuels (Mt)", value=3.0),
        st.number_input("Building Emissions (Mt)", value=7.0),
        st.number_input("CO2 Per Capita", value=1.5),
        st.number_input("Energy Intensity", value=0.8),
        st.number_input("Land-Use Change and Forestry (Mt)", value=0.0)
    ]

    # Predict Carbon Emissions
    if st.button("Predict"):
        try:
            prediction = predict_carbon_emission(input_features)
            st.success(f"The predicted total carbon emissions (excluding LUCF) are: {prediction:.2f} Mt")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")

if __name__ == '__main__':
    main()

