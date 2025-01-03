import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
@st.cache
def load_data():
    data = pd.read_csv("water_potability.csv")
    return data

# Load the dataset
data_clean = load_data()

# Train Random Forest Model
@st.cache
def train_model(data):
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X, y)
    return model

# Train the model
model = train_model(data_clean)

# Streamlit Application Layout
st.title("Water Potability Analysis")
st.write("This application allows you to analyze water quality and predict its potability using Random Forest.")

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(data_clean.head())

# Data Visualization
st.header("Data Visualization")
st.write("Explore individual features in the dataset.")

feature = st.selectbox("Select a feature to visualize:", data_clean.columns[:-1])

if feature:
    # Histogram for the selected feature
    st.subheader(f"Distribution of {feature}")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data_clean[feature], kde=True, bins=30, ax=ax, color="blue")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Box Plot for the selected feature
    st.subheader(f"Box Plot of {feature}")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=data_clean[feature], ax=ax, color="orange")
    ax.set_title(f"Box Plot: {feature}")
    st.pyplot(fig)

# Input for Prediction
st.header("Predict Water Potability")
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=400.0, value=200.0, step=0.1)
solids = st.number_input("Solids (mg/L)", min_value=0.0, max_value=100000.0, value=20000.0, step=0.1)
chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=300.0, step=0.1)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=800.0, value=400.0, step=0.1)
organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, max_value=120.0, value=60.0, step=0.1)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=6.0, value=3.0, step=0.1)

# Prediction Button
if st.button("Predict Potability"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(input_data)
    potability = "Potable" if prediction[0] == 1 else "Not Potable"
    st.subheader(f"Prediction: {potability}")

# Footer
st.write("---")
st.write("Developed by Farrel Riyan Wibowo | A11.2021.13776")
