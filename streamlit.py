import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
def load_model():
    model_file = 'random_forest_model.pkl'  # Change to match your model's filename
    if not os.path.exists(model_file):
        st.error("Model file not found. Please ensure 'random_forest_model.pkl' is in the same directory as this script.")
        return None
    return joblib.load(model_file)

model = load_model()

def predict_carbon_emission(population, gdp_per_capita, gdp_per_capita_ppp, area, transportation, other_fuel_combustion,
                            manufacturing, industrial_processes, fugitive_emissions, energy, electricity_heat,
                            bunker_fuels, building, co2_per_capita, energy_intensity, land_use_change_and_forestry):
    """
    Predict total carbon emissions using the pre-trained model.
    """
    # Create input DataFrame with the relevant features
    input_data = pd.DataFrame(
        [[population, gdp_per_capita, gdp_per_capita_ppp, area, transportation, other_fuel_combustion, manufacturing,
          industrial_processes, fugitive_emissions, energy, electricity_heat, bunker_fuels, building, co2_per_capita,
          energy_intensity, land_use_change_and_forestry]],
        columns=['Population', 'GDP PER CAPITA (USD)', 'GDP PER CAPITA PPP (USD)', 'Area (Km2)', 
                 'Transportation (Mt)', 'Other Fuel Combustion (Mt)', 'Manufacturing/Construction (Mt)', 
                 'Industrial Processes (Mt)', 'Fugitive Emissions (Mt)', 'Energy (Mt)', 
                 'Electricity/Heat (Mt)', 'Bunker Fuels (Mt)', 'Building (Mt)', 'CO2 per Capita', 
                 'Energy Intensity', 'Land-Use Change and Forestry (Mt)']
    )
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title('Carbon Emission Prediction')
    st.write('Enter the details below to predict total carbon emissions (excluding LUCF):')

    if model is None:
        st.error("Model could not be loaded. Please check the model file and restart the app.")
        return

    # Input fields for the features in your dataset
    population = st.number_input("Population", min_value=1, value=1000000, step=100000)
    gdp_per_capita = st.number_input("GDP Per Capita (USD)", min_value=0.0, value=2000.0, step=100.0)
    gdp_per_capita_ppp = st.number_input("GDP Per Capita PPP (USD)", min_value=0.0, value=3000.0, step=100.0)
    area = st.number_input("Area (Km2)", min_value=1, value=500000, step=1000)
    transportation = st.number_input("Transportation Emissions (Mt)", min_value=0.0, value=10.0, step=0.1)
    other_fuel_combustion = st.number_input("Other Fuel Combustion (Mt)", min_value=0.0, value=5.0, step=0.1)
    manufacturing = st.number_input("Manufacturing/Construction Emissions (Mt)", min_value=0.0, value=15.0, step=0.1)
    industrial_processes = st.number_input("Industrial Processes Emissions (Mt)", min_value=0.0, value=8.0, step=0.1)
    fugitive_emissions = st.number_input("Fugitive Emissions (Mt)", min_value=0.0, value=6.0, step=0.1)
    energy = st.number_input("Energy Emissions (Mt)", min_value=0.0, value=20.0, step=0.1)
    electricity_heat = st.number_input("Electricity/Heat Emissions (Mt)", min_value=0.0, value=12.0, step=0.1)
    bunker_fuels = st.number_input("Bunker Fuels (Mt)", min_value=0.0, value=3.0, step=0.1)
    building = st.number_input("Building Emissions (Mt)", min_value=0.0, value=7.0, step=0.1)
    co2_per_capita = st.number_input("CO2 Per Capita", min_value=0.0, value=1.5, step=0.1)
    energy_intensity = st.number_input("Energy Intensity", min_value=0.0, value=0.8, step=0.1)
    land_use_change_and_forestry = st.number_input("Land-Use Change and Forestry (Mt)", min_value=0.0, value=0.0, step=0.1)

    # Predict Carbon Emissions
    if st.button("Predict"):
        prediction = predict_carbon_emission(
            population, gdp_per_capita, gdp_per_capita_ppp, area, transportation, other_fuel_combustion, 
            manufacturing, industrial_processes, fugitive_emissions, energy, electricity_heat, 
            bunker_fuels, building, co2_per_capita, energy_intensity, land_use_change_and_forestry
        )
        st.success(f"The predicted total carbon emissions (excluding LUCF) are: {prediction:.2f} Mt")

if __name__ == '__main__':
    main()
