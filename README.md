# 🧠 NeuroChurn

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

<p align="center">
  <b>Deep Learning Predictive Customer Retention & Attrition Intelligence Engine</b><br>
  <i>Empowering fintech and banking institutions to proactively detect, analyze, and prevent customer churn using Artificial Neural Networks.</i>
</p>

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Specifications](#-model-specifications)
- [Dataset & Telemetry](#-dataset--telemetry)
- [Installation & Quickstart](#-installation--quickstart)
- [Project Structure](#-project-structure)
- [Interactive Dashboard](#-interactive-dashboard)
- [Author & Acknowledgments](#-author--acknowledgments)
- [License](#-license)

---

## 💡 Overview

Customer attrition poses a multimillion-dollar threat across retail banking and subscription enterprises. **NeuroChurn** addresses this challenge by deploying a fine-tuned **Artificial Neural Network (ANN)** that continuously assesses customer churn probability based on historical financial behaviors, activity levels, credit ratings, and demographic metrics.

Rather than relying on reactive surveys, NeuroChurn enables customer success teams to intervene **before** account cancellation takes place.

---

## ⚡ Key Features

- **⚡ Real-Time Neural Inference**: Sub-second churn probability evaluation powered by TensorFlow/Keras runtime.
- **🎨 Glassmorphic Modern Dashboard**: Highly responsive, dark-mode analytical interface built with Streamlit.
- **🔄 Robust Preprocessing Pipeline**: Standalone serialized pipelines for categorical encoding (`OneHotEncoder`, `LabelEncoder`) and numeric standard scaling (`StandardScaler`).
- **🛡️ Risk Stratification & Actionable Alerts**: Automatic categorization into high-risk and low-risk cohorts with tailored mitigation recommendations.
- **📊 Comprehensive Telemetry Cards**: Instant visual breakdown of tenure, credit score, balance-to-salary ratios, and engagement factors.

---

## 🏗️ System Architecture

```
[ Customer Telemetry Data ]
            │
            ▼
┌───────────────────────────────────────────────┐
│        Feature Preprocessing Pipeline         │
│  - Categorical Encoding (Geography & Gender)  │
│  - Continuous Normalization (StandardScaler)  │
└───────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────┐
│     Artificial Neural Network (ANN) Engine    │
│  - Input Layer: 12 Normalized Features        │
│  - Dense Hidden Layers with ReLU Activation   │
│  - Output Layer: Sigmoid (Binary Churn Proba) │
└───────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────┐
│     Streamlit Real-Time Decision Portal       │
│  - Dynamic Churn Gauge & Metric Cards         │
│  - Proactive Retention Advisory Triggers      │
└───────────────────────────────────────────────┘
```

---

## 🧠 Model Specifications

| Attribute | Specification |
| :--- | :--- |
| **Model Type** | Artificial Neural Network (Feedforward ANN) |
| **Framework** | TensorFlow 2.15 / Keras |
| **Input Dimensions** | 12 features (Demographic + Financial + Encoded Geography) |
| **Hidden Layer Activation** | Rectified Linear Unit (ReLU) |
| **Output Activation** | Sigmoid ($\sigma(z) = rac{1}{1 + e^{-z}}$) |
| **Loss Function** | Binary Cross-Entropy |
| **Optimization** | Adam Optimizer |
| **Primary Evaluation Metric** | Accuracy (~86%) & ROC-AUC |

---

## 📊 Dataset & Telemetry

The model is trained on banking customer records comprising the following attributes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `CreditScore` | Integer | Customer credit evaluation score (300 - 850) |
| `Geography` | Categorical | Country of residence (`France`, `Germany`, `Spain`) |
| `Gender` | Categorical | Customer gender (`Female`, `Male`) |
| `Age` | Integer | Customer age (18 - 92) |
| `Tenure` | Integer | Number of years as an active bank member |
| `Balance` | Float | Account balance |
| `NumOfProducts` | Integer | Quantity of banking products subscribed (1 - 4) |
| `HasCrCard` | Binary | Credit card possession (`1` = Yes, `0` = No) |
| `IsActiveMember` | Binary | Customer engagement status (`1` = Active, `0` = Inactive) |
| `EstimatedSalary` | Float | Estimated annual salary |

---

## 🚀 Installation & Quickstart

### Prerequisites

- **Python 3.10** (Recommended)
- **Git**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ayush-216/NeuroChurn.git
cd NeuroChurn
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
.\venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Streamlit Application

```bash
streamlit run app.py
```

The web dashboard will automatically launch at `http://localhost:8501`.

---

## 📂 Project Structure

```
NeuroChurn/
├── dataset/
│   └── Churn_Modelling.csv     # 10,000 banking customer telemetry records
├── app.py                      # Real-time Streamlit dashboard application
├── model.h5                    # Production-trained Neural Network model weights
├── scaler.pkl                  # Serialized StandardScaler for input features
├── encoder.pkl                 # OneHotEncoder for Geography categories
├── encodegender.pkl            # LabelEncoder for Gender categories
├── Raw.ipynb                   # Exploratory data analysis & feature distributions
├── prediction.ipynb            # Deep neural network training and validation pipeline
├── requirements.txt            # Python package dependencies
├── runtime.txt                 # Runtime environment definition (Python 3.10)
├── .gitignore                  # Git tracking exclusion filters
└── README.md                   # Project documentation and architectural manual
```

---

## 🖥️ Interactive Dashboard

The NeuroChurn dashboard provides:
1. **Interactive Sidebar Telemetry**: Dial in customer demographics, balance, credit score, tenure, and product counts.
2. **Profile Breakdown Card**: Clean visual ledger summarizing all submitted parameters before inference.
3. **Instant Risk Assessment**: Immediate classification with high/low risk indicators and recommended operational next steps.

---

## 👨‍💻 Author

**Ayush**  
- **GitHub**: [@Ayush-216](https://github.com/Ayush-216)  
- **Email**: [exploretheuniverse216@gmail.com](mailto:exploretheuniverse216@gmail.com)

---

## 📜 License

This project is open-source and distributed under the [MIT License](LICENSE).
