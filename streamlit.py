import streamlit as st
import requests

st.title("H1B Visa Case Status Prediction")

# Input fields
continent = st.selectbox("Continent", ['Asia', 'Africa', 'North America', 'Europe', 'South America', 'Oceania'])
education = st.selectbox("Education Level", ['High School', "Master's", "Bachelor's", 'Doctorate'])
job_exp = st.radio("Has Job Experience?", ['Y', 'N'])
job_training = st.radio("Requires Job Training?", ['Y', 'N'])
region = st.selectbox("Region of Employment", ['West', 'Northeast', 'South', 'Midwest', 'Island'])
wage_unit = st.selectbox("Unit of Wage", ['Hour', 'Year', 'Week', 'Month'])
full_time = st.radio("Full-time Position?", ['Y', 'N'])

no_of_employees = st.number_input("Number of Employees", min_value=0)
prevailing_wage = st.number_input("Prevailing Wage", min_value=0.0)
company_age = st.number_input("Company Age (years)", min_value=0)

# Construct input data (no list wrapping!)
input_dict = {
    'continent': continent,
    'education_of_employee': education,
    'has_job_experience': job_exp,
    'requires_job_training': job_training,
    'region_of_employment': region,
    'unit_of_wage': wage_unit,
    'full_time_position': full_time,
    'no_of_employees': no_of_employees,
    'prevailing_wage': prevailing_wage,
    'company_age': company_age
}

if st.button("Predict Case Status"):
    response = requests.post("http://localhost:8000/predict", json=input_dict)

    if response.status_code == 200:
        result = response.json()
        st.write("Prediction:", "✅ Certified" if result['prediction'] == "Certified" else "❌ Denied")
        if 'probability' in result:
            st.write(f"Probability of Certification: {result['probability']:.2%}")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
