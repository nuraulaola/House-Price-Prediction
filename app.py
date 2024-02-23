import os
import requests
import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# ... (rest of your code)

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
    user_input = get_user_input(maximal_values)

    # Transform categorical variables using label encoders
    for col in user_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Make Ridge Regression Prediction
    prediction = Ridge_Model.predict(user_input)

    # Display Ridge prediction
    st.write(f"Predicted Price (Ridge Regression): {int(prediction[0])}")

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
    st.write(f"Predicted Price (Lasso Regression): {int(prediction[0])}")

    # Show toast notification
    st.toast('Lasso Regression prediction complete!', icon='😊')

def get_user_input(maximal_values):
    # ... (rest of your code)

if __name__ == "__main__":
    main()
