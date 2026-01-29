# 🧠 NeuroChurn — Customer Churn Prediction Engine

NeuroChurn is a deep learning-powered predictive intelligence platform designed to forecast customer attrition before it happens. Built with TensorFlow and Streamlit, it transforms multidimensional customer banking telemetry into actionable retention insights.

---

## 📌 Architecture & System Flow

```
Raw Customer Telemetry
        │
        ▼
Data Preprocessing Pipeline (StandardScaler + OneHotEncoder)
        │
        ▼
Artificial Neural Network (TensorFlow / Keras)
        │
        ▼
Real-Time Streamlit Decision Dashboard
```

## ⚙️ Quickstart

```bash
git clone https://github.com/Ayush-216/NeuroChurn.git
cd NeuroChurn
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
