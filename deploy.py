import streamlit as st
import pandas as pd
import joblib
import locale
import os
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Housing Market Forecast",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

model_path = r'model.pkl'
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}")
except EOFError:
    st.error("Error loading model. The file might be corrupted or incomplete.")

# Load the dataset to get feature names and encoders
file_path = r'test.csv'
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
else:
    st.error(f"Data file not found at {file_path}")

# Extract the feature names
feature_names = data.drop(columns=['Id']).columns.tolist()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Custom CSS for styling and background image
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://imgur.com/a/housing-market-forecast-MmETowm');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stButton > button {
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput > label {
        font-weight: bold;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .stNumberInput > label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Housing Market Forecast")

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        input_data[feature] = st.selectbox(f"Select {feature}", options)
    else:
        avg_value = data[feature].mean()
        input_data[feature] = st.number_input(f"Enter {feature}", value=avg_value)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features
for feature in label_encoders:
    input_df[feature] = label_encoders[feature].transform(input_df[feature])

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        predicted_usd = prediction[0]

        # Conversion rate (1 USD to INR)
        conversion_rate = 82.50  # Update based on the latest rate
        predicted_inr = predicted_usd * conversion_rate

        # Set locale to Indian
        locale.setlocale(locale.LC_ALL, 'en_IN')

        # Format the predicted values
        formatted_usd = f"${predicted_usd:,.2f}"
        formatted_inr = locale.format_string("%0.2f", predicted_inr, grouping=True)

        st.write(f"Estimated Cost in USD: {formatted_usd}")
        st.write(f"Estimated Cost in INR: ₹{formatted_inr}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")