!pip install streamlit

import requests
import pickle
import pandas as pd
import streamlit as st

# Load label encoders
label_encoders_url = 'https://github.com/nuraulaola/House-Price-Prediction/raw/main/Label_Encoders.pkl'
label_encoders = pickle.loads(requests.get(label_encoders_url).content)

# Load Ridge Regression model
ridge_model_url = 'https://github.com/nuraulaola/House-Price-Prediction/raw/main/ridge_reg_model.pkl'
ridge_model = pickle.loads(requests.get(ridge_model_url).content)

# Load Lasso Regression model
lasso_model_url = 'https://github.com/nuraulaola/House-Price-Prediction/raw/main/lasso_reg_model.pkl'
lasso_model = pickle.loads(requests.get(lasso_model_url).content)

# Streamlit app
def main():
    st.title("House Price Prediction App")

    # Sidebar to choose the model
    model_choice = st.sidebar.radio("Select Model", ("Ridge Regression", "Lasso Regression"))

    if model_choice == "Ridge Regression":
        predict_ridge()
    elif model_choice == "Lasso Regression":
        predict_lasso()

def predict_ridge():
    st.header("Ridge Regression Prediction")

    # Get user input
    user_input = get_user_input()

    # Transform categorical variables using label encoders
    for col in user_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Make Ridge Regression Prediction
    prediction = ridge_model.predict(user_input)

    # Display Ridge prediction
    st.write(f"Predicted Price (Ridge Regression): {int(prediction[0])}")

def predict_lasso():
    st.header("Lasso Regression Prediction")

    # Get user input
    user_input = get_user_input()

    # Transform categorical variables using label encoders
    for col in user_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Make Lasso Regression Prediction
    prediction = lasso_model.predict(user_input)

    # Display Lasso prediction
    st.write(f"Predicted Price (Lasso Regression): {int(prediction[0])}")

def get_user_input():
    # Create input form using Streamlit
    st.sidebar.header('User Input Parameters')

    # Add input elements for the user to enter data
    made = st.sidebar.number_input("Enter Year Made:", min_value=1800, max_value=2022, value=2000)
    square_meters = st.sidebar.number_input("Enter Square Meters:", min_value=0.0, value=100.0)
    number_of_rooms = st.sidebar.number_input("Enter Number of Rooms:", min_value=0, value=3)
    has_storage_room = st.sidebar.checkbox("Has Storage Room")
    has_guest_room = st.sidebar.checkbox("Has Guest Room")
    # Add more input elements for other features as needed

    # Create a dictionary with user input data
    input_data = {
        "made": made,
        "squareMeters": square_meters,
        "numberOfRooms": number_of_rooms,
        "hasStorageRoom": has_storage_room,
        "hasGuestRoom": has_guest_room,
        # Add more key-value pairs for other features
    }

    # Return the user input as a DataFrame
    return pd.DataFrame([input_data])

if __name__ == "__main__":
    main()
