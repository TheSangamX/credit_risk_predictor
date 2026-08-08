# 💳 Credit Risk Predictor

An end-to-end Machine Learning project that predicts the probability of loan default using applicant financial, demographic, and credit-related information.

The project covers data analysis, preprocessing, model training, evaluation, feature importance analysis, and an interactive **Streamlit** web application for real-time credit risk prediction.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Machine Learning Model](#-machine-learning-model)
- [Model Performance](#-model-performance)
- [Important Features](#-important-features)
- [Streamlit Application](#️-streamlit-application)
- [Project Structure](#-project-structure)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Technologies Used](#️-technologies-used)
- [Installation](#️-installation)
- [Run the Application](#️-run-the-application)
- [Example Prediction](#-example-prediction)
- [Feature Importance](#-feature-importance)
- [Important Note](#️-important-note)
- [Project Highlights](#-project-highlights)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Project Overview

Credit risk assessment is a critical task for financial institutions, since incorrectly approving a high-risk applicant can result in significant financial losses.

This project uses Machine Learning to estimate the **probability that an applicant may default on a loan**.

The final model is integrated into a Streamlit application where users can enter applicant information and instantly receive:

- ✅ Predicted default probability
- ✅ Credit risk classification
- ✅ Decision threshold comparison
- ✅ Model performance metrics
- ✅ Top 20 important features

---

## 🧠 Machine Learning Model

**Model used:** `HistGradientBoostingClassifier`

The trained preprocessing pipeline and model are serialized using `joblib`. The Streamlit application loads this saved pipeline and applies the exact same preprocessing used during training before generating predictions — ensuring consistency between training and inference.

---

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 85.17% |
| Precision | 24.98% |
| Recall | 41.79% |
| F1 Score | 31.27% |
| ROC-AUC | 75.99% |

### Decision Threshold

The model uses a custom decision threshold of **0.15 (15%)** instead of the default 0.50, since this threshold is more appropriate for the credit-risk classification task.

| Predicted Probability | Classification |
|---|---|
| ≥ 15% | 🔴 Higher Credit Risk |
| < 15% | 🟢 Lower Credit Risk |

---

## 🔍 Important Features

The Top 20 features that most strongly influenced the final model's predictions:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | External Source 3 | 0.062478 |
| 2 | External Source 2 | 0.048455 |
| 3 | Goods Price | 0.019804 |
| 4 | External Source 1 | 0.019637 |
| 5 | Credit Amount | 0.012610 |
| 6 | Age | 0.006826 |
| 7 | Loan Annuity | 0.004238 |
| 8 | Days Employed | 0.003826 |
| 9 | Gender: Female | 0.002707 |
| 10 | Education: Higher Education | 0.002321 |
| 11 | Owns Car: No | 0.002181 |
| 12 | Days Since ID Published | 0.001773 |
| 13 | Family Status: Married | 0.001443 |
| 14 | Contract Type: Cash Loan | 0.001372 |
| 15 | Days Since Last Phone Change | 0.001181 |
| 16 | Credit Bureau Requests: Quarterly | 0.000878 |
| 17 | Gender: Male | 0.000840 |
| 18 | Document 3 Flag | 0.000743 |
| 19 | Income Type: Working | 0.000685 |
| 20 | Days Since Registration | 0.000610 |

---

## 🖥️ Streamlit Application

An interactive web application built with Streamlit powers real-time predictions.

### Input Categories

| Category | Fields |
|---|---|
| 👤 **Personal Information** | Gender, Age, Family Status, Education |
| 💰 **Financial Information** | Annual Income, Credit Amount, Loan Annuity, Goods Price |
| 🏠 **Applicant Profile** | Owns Car, Owns Property, Income Type, Housing Type |

### Prediction Output

The application displays:

- Predicted Default Probability
- Credit Risk Classification
- Decision Threshold
- Model Performance
- Top 20 Feature Importance Chart
- Feature Importance Details

---

## 📁 Project Structure

```text
credit-risk-ml/
│
├── data/
│   └── application_train.csv
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── models/
│   ├── credit_risk_model.joblib
│   └── app_defaults.joblib
│
├── src/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Numerical Feature Processing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Threshold Selection
     ↓
Feature Importance Analysis
     ↓
Model Serialization
     ↓
Streamlit Application
     ↓
Credit Risk Prediction
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Programming Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, HistGradientBoostingClassifier |
| **Model Persistence** | Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Development Tools** | Jupyter Notebook, Visual Studio Code, Git, GitHub |

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/TheSangamX/credit_risk_predictor.git>
   ```

2. **Navigate to the project directory:**
   ```bash
   cd credit-risk-ml
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   ```

   *Windows:*
   ```cmd
   venv\Scripts\activate
   ```

   *macOS/Linux:*
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:
`http://localhost:8501`

---

## 📌 Example Prediction

**Low risk example:**

```text
Predicted Default Probability: 4.34%

🟢 LOWER CREDIT RISK
Predicted probability is below the 15% decision threshold.
```

**High risk example:**

```text
🔴 HIGH CREDIT RISK
Predicted probability is at or above the 15% decision threshold.
```

---

## 📈 Feature Importance

The application also provides a visual representation of the Top 20 features that contributed most strongly to the final model's predictions, helping users understand which variables were most influential in generating each prediction.

---

## ⚠️ Important Note

This project is developed for **educational and portfolio purposes only**.

The predictions generated by this application should **not** be considered professional financial or lending advice. A production-grade credit-risk system would require additional validation, regulatory compliance, fairness analysis, model monitoring, explainability, and domain-specific risk controls.

---

## 🎯 Project Highlights

- End-to-end Machine Learning workflow
- Data preprocessing pipeline
- HistGradientBoosting classification model
- Probability-based credit risk prediction
- Custom decision threshold
- Model evaluation using multiple classification metrics
- Feature importance analysis
- Saved ML pipeline using Joblib
- Interactive Streamlit dashboard
- Real-time prediction
- Portfolio-ready project structure

---

## ⭐ Future Improvements

- SHAP-based individual prediction explanations
- Model calibration
- Hyperparameter optimization
- Cross-validation
- Fairness and bias analysis
- Automated model monitoring
- Docker deployment
- Cloud deployment
- REST API integration
- Database integration

---

## 👨‍💻 Author

**Sangam Gupta**

Computer Engineering | Data Science & Machine Learning

**Interested in:**
- Artificial Intelligence
- Machine Learning
- Data Science
- AI Engineering
- Quantitative Finance
