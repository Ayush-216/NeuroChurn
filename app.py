import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="NeuroChurn | AI Customer Retention Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.95rem;
    }
    
    .data-label {
        color: #cbd5e1;
    }
    
    .data-value {
        font-weight: 600;
        color: #38bdf8;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .risk-low {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('encoder.pkl', 'rb') as f:
        geo_encoder = pickle.load(f)
    with open('encodegender.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, geo_encoder, gender_encoder, scaler

model, oheencodergeo, labelencodergender, scaler = load_assets()

# Header
st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
        <h1 style='margin-bottom: 0.2rem; font-size: 2.4rem;'>🧠 NeuroChurn</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0;'>
            Deep Learning Predictive Customer Retention & Attrition Intelligence Engine
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar input controls
st.sidebar.markdown("### 👤 Customer Profile Telemetry")
st.sidebar.markdown("Configure demographic and account metrics:")

geography = st.sidebar.selectbox('Geography', oheencodergeo.categories_[0], help="Customer's geographic location")
gender = st.sidebar.selectbox('Gender', labelencodergender.classes_)
age = st.sidebar.slider('Age', 18, 92, 38)
tenure = st.sidebar.slider('Tenure (Years with Bank)', 0, 10, 5)
num_of_products = st.sidebar.slider('Number of Products Subscribed', 1, 4, 2)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.sidebar.selectbox('Active Membership Status', [1, 0], format_func=lambda x: "Active Member" if x == 1 else "Inactive Member")

st.sidebar.markdown("### 💰 Financial Parameters")
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
balance = st.sidebar.number_input('Account Balance ($)', min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0)
estimated_salary = st.sidebar.number_input('Estimated Annual Salary ($)', min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)

# Layout
col_summary, col_prediction = st.columns([1.3, 1.0], gap="large")

with col_summary:
    st.markdown("### 📋 Customer Profile Overview")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Account & Demographic Breakdown</div>
        <div class="data-row"><span class="data-label">Geography</span><span class="data-value">{geography}</span></div>
        <div class="data-row"><span class="data-label">Gender</span><span class="data-value">{gender}</span></div>
        <div class="data-row"><span class="data-label">Age</span><span class="data-value">{age} yrs</span></div>
        <div class="data-row"><span class="data-label">Credit Score</span><span class="data-value">{credit_score}</span></div>
        <div class="data-row"><span class="data-label">Account Balance</span><span class="data-value">${balance:,.2f}</span></div>
        <div class="data-row"><span class="data-label">Tenure</span><span class="data-value">{tenure} yrs</span></div>
        <div class="data-row"><span class="data-label">Active Products</span><span class="data-value">{num_of_products}</span></div>
        <div class="data-row"><span class="data-label">Credit Card Holder</span><span class="data-value">{'Yes' if has_cr_card else 'No'}</span></div>
        <div class="data-row"><span class="data-label">Membership Status</span><span class="data-value">{'Active' if is_active_member else 'Inactive'}</span></div>
        <div class="data-row"><span class="data-label">Estimated Salary</span><span class="data-value">${estimated_salary:,.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    predict_button = st.button("⚡ Run Neural Inference")

with col_prediction:
    st.markdown("### 📊 Churn Probability Analysis")
    
    if predict_button:
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
        
        prediction = model.predict(input_data_scaled, verbose=0)
        churn_proba = float(prediction[0][0])
        retention_proba = 1.0 - churn_proba
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Confidence Metrics</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {'#ef4444' if churn_proba > 0.5 else '#22c55e'}; margin: 10px 0;">
                {churn_proba * 100:.1f}% <span style="font-size: 1rem; color: #94a3b8; font-weight: normal;">churn probability</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="font-size: 0.85rem; color: #cbd5e1;">Retention Likelihood: <b>{retention_proba * 100:.1f}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(churn_proba)
        
        if churn_proba >= 0.50:
            st.markdown(f"""
            <div class="risk-high">
                ⚠️ <b>High Risk of Attrition</b><br>
                <small style="font-weight: 400; color: #fca5a5;">
                    Immediate retention campaign recommended. Consider offering product discounts, personalized relationship management, or premium credit incentives.
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                ✅ <b>Low Risk Customer</b><br>
                <small style="font-weight: 400; color: #86efac;">
                    Account behavior aligns with loyal user cohort. Prime candidate for cross-selling and premium tiers.
                </small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Adjust customer telemetry on the sidebar and click **'Run Neural Inference'** to evaluate retention risk.")
