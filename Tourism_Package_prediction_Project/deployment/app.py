import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Niraj8767/Tourism-Package-Prediction-MLOPs-Model", filename="tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase a Wellness Tourism Package.
Fill in the customer details below to get a prediction.
""")

# User input fields

# Categorical features
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Sector'])
gender = st.selectbox("Gender", ['Male', 'Female'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Super Deluxe', 'Standard', 'King'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])

# Numerical features
city_tier = st.number_input("City Tier (1, 2, or 3)", min_value=1, max_value=3, value=1, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2, step=1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
preferred_property_star = st.number_input("Preferred Property Star (3, 4, or 5)", min_value=3.0, max_value=5.0, value=3.0, step=1.0)
number_of_trips = st.number_input("Number of Trips Annually", min_value=0.0, max_value=30.0, value=2.0, step=1.0)
passport = st.selectbox("Has Passport? (0=No, 1=Yes)", [0, 1])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, step=1)
own_car = st.selectbox("Owns a Car? (0=No, 1=Yes)", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, value=1.0, step=1.0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=20000.0, step=100.0)
age = st.slider("Age", min_value=18, max_value=70, value=35)

# Preprocess Age to AgeGroup, matching prep.py logic
def get_age_group(age):
    if age < 18:
        return '<18'
    elif 18 <= age <= 30:
        return '18-30'
    elif 31 <= age <= 45:
        return '31-45'
    elif 46 <= age <= 60:
        return '46-60'
    else:
        return '>60'

age_group = get_age_group(age)

# Assemble input into DataFrame
# Ensure the order of columns matches the training data features
input_data = pd.DataFrame([{
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'AgeGroup': age_group # Add AgeGroup here
}])

if st.button("Predict Purchase"):    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[:, 1][0] # Probability of positive class
    
    if prediction == 1:
        st.success(f"Prediction: Customer is likely to purchase the package (Probability: {probability:.2f})")
    else:
        st.info(f"Prediction: Customer is unlikely to purchase the package (Probability: {probability:.2f})")
