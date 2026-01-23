import streamlit as st
import pandas as pd

st.set_page_config(page_title="NeuroChurn", page_icon="🧠", layout="wide")

st.title("🧠 NeuroChurn: Customer Churn Intelligence")
st.markdown("---")

st.sidebar.header("Customer Profile")
geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
age = st.sidebar.slider('Age', 18, 92, 35)
balance = st.sidebar.number_input('Balance', value=50000.0)
credit_score = st.sidebar.number_input('Credit Score', value=650)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000.0)
tenure = st.sidebar.slider('Tenure', 0, 10, 3)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [1, 0])
is_active_member = st.sidebar.selectbox('Is Active Member', [1, 0])

st.write("Configured customer telemetry:")
st.json({
    "geography": geography,
    "gender": gender,
    "age": age,
    "balance": balance,
    "credit_score": credit_score,
    "estimated_salary": estimated_salary,
    "tenure": tenure,
    "num_of_products": num_of_products,
    "has_cr_card": has_cr_card,
    "is_active_member": is_active_member
})
