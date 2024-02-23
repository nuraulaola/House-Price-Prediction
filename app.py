import os
import requests
import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a Bug': 'https://www.example.com/bug',
        'About': "# House Price Prediction App\n\nThis app predicts house prices using Ridge and Lasso regression models."
    }
)

with open('Label_Encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

with open('ridge_reg_model.pkl', 'rb') as file:
    Ridge_Model = pickle.load(file)

with open('lasso_reg_model.pkl', 'rb') as file:
    Lasso_Model = pickle.load(file)

# Maximal values for input columns
maximal_values = {
    "made": 2022,
    "squareMeters": 5000.0,
    "numberOfRooms": 6,
}

# Streamlit app
def main():
    st.title("House Price Prediction App")

    # Sidebar to choose the model
    model_choice = st.sidebar.radio("Select Model", ("Ridge Regression", "Lasso Regression"))

    if model_choice == "Ridge Regression":
        st.header("Ridge Regression Prediction")
        st.markdown("Ridge Regression helps our app predict house prices by keeping things simple and preventing"
                    " the computer from getting too complicated. 🏠💡")

        predict_ridge()
    elif model_choice == "Lasso Regression":
        st.header("Lasso Regression Prediction")
        st.markdown("Lasso Regression is another way our app predicts house prices. It focuses on the most important"
                    " things, making it easy to understand and keeping things clear. 🏡✨")

        predict_lasso()

def predict_ridge():
    st.header("Ridge Regression Prediction")

    # Get user input
    user_input = get_user_input(maximal_values)

    # Transform categorical variables using label encoders
    for col in user_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Make Ridge Regression Prediction
    prediction = Ridge_Model.predict(user_input)

    # Display Ridge prediction
    st.markdown(f"Predicted Price (Ridge Regression): **{int(prediction[0])}**")

    # Show toast notification
    st.toast('Ridge Regression prediction complete!', icon='😊')

def predict_lasso():
    st.header("Lasso Regression Prediction")

    # Get user input
    user_input = get_user_input(maximal_values)

    # Transform categorical variables using label encoders
    for col in user_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Make Lasso Regression Prediction
    prediction = Lasso_Model.predict(user_input)

    # Display Lasso prediction
    st.markdown(f"Predicted Price (Lasso Regression): **{int(prediction[0])}**")

    # Show toast notification
    st.toast('Lasso Regression prediction complete!', icon='😊')

def get_user_input(maximal_values):
    # Create input form using Streamlit
    st.sidebar.header('User Input Parameters')

    # Add input elements for the user to enter data
    made = st.sidebar.slider("Enter Year Made 🏠:", min_value=1800, max_value=maximal_values["made"], value=2000)
    square_meters = st.sidebar.slider("Enter Square Meters 📏:", min_value=0.0, max_value=maximal_values["squareMeters"], value=100.0)
    number_of_rooms = st.sidebar.slider("Enter Number of Rooms 🛏️:", min_value=0, max_value=maximal_values["numberOfRooms"], value=3)
    has_storage_room = st.sidebar.checkbox(":package: Has Storage Room")
    has_guest_room = st.sidebar.checkbox(":busts_in_silhouette: Has Guest Room")

    # Create a dictionary with user input data
    input_data = {
        "made": made,
        "squareMeters": square_meters,
        "numberOfRooms": number_of_rooms,
        "hasStorageRoom": has_storage_room,
        "hasGuestRoom": has_guest_room,
    }

    # Return the user input as a DataFrame
    return pd.DataFrame([input_data])

if __name__ == "__main__":
    main()
