import streamlit as st

# Import the pages directly
from enter_data import main as enter_data_main  # Import the main function from enter_data.py
from outcome import main as outcome_main  # Import the main function from outcome.py

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["ECG Data Upload", "Outcome Prediction"])

# Display the selected page
if page == "ECG Data Upload":
    enter_data_main()  # Call the main function from enter_data.py
elif page == "Outcome Prediction":
    outcome_main()  # Call the main function from outcome.py