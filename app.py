import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(page_title="NeuroChurn", page_icon="🧠", layout="wide")

model = tf.keras.models.load_model('model.h5')
with open('encoder.pkl', 'rb') as f:
    oheencodergeo = pickle.load(f)
with open('encodegender.pkl', 'rb') as f:
    labelencodergender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🧠 NeuroChurn: Customer Churn Intelligence")
st.markdown("---")

st.sidebar.header("Customer Telemetry")
geography = st.sidebar.selectbox('Geography', oheencodergeo.categories_[0])
gender = st.sidebar.selectbox('Gender', labelencodergender.classes_)
age = st.sidebar.slider('Age', 18, 92, 38)
balance = st.sidebar.number_input('Balance ($)', value=60000.0)
credit_score = st.sidebar.number_input('Credit Score', value=650)
estimated_salary = st.sidebar.number_input('Estimated Salary ($)', value=50000.0)
tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 5)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 2)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [1, 0])
is_active_member = st.sidebar.selectbox('Is Active Member', [1, 0])

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Customer Parameters Summary")
    st.write(f"**Geography:** {geography} | **Gender:** {gender} | **Age:** {age}")
    st.write(f"**Credit Score:** {credit_score} | **Balance:** ${balance:,.2f} | **Salary:** ${estimated_salary:,.2f}")
    st.write(f"**Tenure:** {tenure} yrs | **Products:** {num_of_products}")

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelencodergender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = oheencodergeo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oheencodergeo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_proba = float(prediction[0][0])
    
    with col2:
        st.subheader("Prediction Result")
        st.progress(churn_proba)
        st.metric("Churn Probability", f"{churn_proba:.2%}")
        
        if churn_proba > 0.5:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")
