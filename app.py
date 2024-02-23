import os
import requests
import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction Assistant App ⋆｡ﾟ☁︎",
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
    st.title("House Price Prediction Assitant App")

    # Sidebar to choose the page
    selected_page = st.sidebar.selectbox("Select Page", ["Home", "Ridge Regression", "Lasso Regression"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "Ridge Regression":
        ridge_page()
    elif selected_page == "Lasso Regression":
        lasso_page()

def home_page():
    st.write("Welcome to House Price Prediction App!")
    st.write("This app predicts house prices using Ridge and Lasso regression models.")

def ridge_page():
    st.header("Ridge Regression Prediction")
    st.markdown("Ridge Regression helps our app predict house prices by keeping things simple and preventing"
                " the computer from getting too complicated. 🏠💡")

    prediction = predict_ridge()

    # Display Ridge prediction in a floating box
    st.success(f"🚀 **Great News!** Our magic model has predicted the house price using Ridge Regression. Based on the"
               f" provided details, the estimated price is **${int(prediction)}**. 🏠✨")

    # Show toast notification
    st.toast('Ridge Regression prediction complete!', icon='😊')

def lasso_page():
    st.header("Lasso Regression Prediction")
    st.markdown("Lasso Regression is another way our app predicts house prices. It focuses on the most important"
                " things, making it easy to understand and keeping things clear. 🏡✨")

    prediction = predict_lasso()

    # Display Lasso prediction in a floating box
    st.info(f"🚀 **Exciting News!** Our advanced model has made a prediction using Lasso Regression. Based on the"
            f" information provided, the predicted house price is **${int(prediction)}**. 🏡✨")

    # Show toast notification
    st.toast('Lasso Regression prediction complete!', icon='😊')

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

    return prediction[0]
    
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

    return prediction[0]

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
