# Predictive Modeling for Early Diagnosis of Alzheimer's Disease

![Project Status](https://img.shields.io/badge/status-in_progress-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006699.svg?logo=xgboost&logoColor=white)

This project aims to develop machine learning models for the early diagnosis of Alzheimer's disease (AD) using structured biomedical data. By leveraging demographic, clinical, behavioral, and cognitive features, we build a predictive system to classify patients as either "Cognitively Normal" or having "Alzheimer's Disease."

## 🧠 Motivation: Why Early Diagnosis Matters

Alzheimer's is a progressive neurodegenerative disorder affecting over 55 million people worldwide, with cases projected to triple by 2050. Early diagnosis is critical for:
- **Preserving Quality of Life:** Early interventions can slow disease progression.
- **Reducing Healthcare Costs:** Late-stage care is significantly more expensive.
- **Improving Outcomes:** Timely treatment leads to a better response to therapies.
- **Supporting Families:** Allows for emotional, financial, and medical preparation.
- **Driving Research:** Accelerates drug discovery and clinical trials.

## 🎯 Problem Statement

The primary challenge in diagnosing Alzheimer's lies in the siloed and inconsistent nature of patient data across different systems (EHRs, labs, imaging). This lack of structured integration and interoperability makes it difficult to build a comprehensive patient profile, leading to:
- Delayed diagnosis
- Worse patient outcomes
- Higher healthcare costs
- Increased caregiver burden

Our goal is to address this gap by developing robust predictive models using integrated, structured biomedical data for early detection.

## 🗂️ Dataset Overview

This project utilizes a dataset containing 2,149 patient records with 35 features related to Alzheimer's disease diagnosis and associated risk factors.

-   **Source:** Internal compilation (`Alzheimer’s Disease Data.xlsx`), inspired by open-access medical datasets such as ADNI and OASIS.
-   **Records:** 2,149
-   **Features:** 35
-   **Target Variable:** `Diagnosis`
    -   `0` → Cognitively Normal (CN)
    -   `1` → Alzheimer’s Disease (AD)

### Main Feature Categories:
-   **Demographics:** Age, Gender, Ethnicity, Education Level
-   **Lifestyle Factors:** Smoking, Alcohol Consumption, Physical Activity, Diet Quality
-   **Medical History:** Family History of Alzheimer's, Cardiovascular Disease, Diabetes, Depression
-   **Clinical Measures:** Blood Pressure (Systolic/Diastolic), Cholesterol Levels, BMI
-   **Cognitive Assessments:** MMSE (Mini-Mental State Exam), Functional Assessment
-   **Behavioral Symptoms:** Memory Complaints, Confusion, Disorientation, Forgetfulness

## ⚙️ Project Workflow

Our methodology follows a standard data science pipeline:

1.  **Data Preparation & Preprocessing:** Cleaning the dataset, handling missing values, encoding categorical variables, and normalizing numerical features.
2.  **Exploratory Data Analysis (EDA):** Analyzing data distributions, correlations, and patterns using visualizations to uncover key insights.
3.  **Model Selection & Building:** Choosing candidate models based on literature and problem type, including Random Forest, XGBoost, Support Vector Machines (SVM), and Logistic Regression.
4.  **Model Training & Testing:** Splitting the data into training and testing sets and evaluating model performance using metrics such as Accuracy, F1-Score, and ROC-AUC.
5.  **Proposed Solution:** Deploying the best-performing model as part of a predictive system designed to assist clinicians in early disease detection.

## 💻 Technologies Used
-   **Data Processing & Analytics:** Pandas, NumPy, Scikit-learn
-   **Machine Learning & Modeling:** Scikit-learn, XGBoost, TensorFlow, PyTorch
-   **Data Visualization:** Matplotlib, Seaborn
-   **Development Environment:** Jupyter Notebook

## 🚀 Getting Started

To run this project locally, follow the steps below.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch Jupyter Notebook:**
```bash
jupyter notebook
```
Open the main notebook file to explore the analysis and models.

## 📚 References

This work is informed by established research in the field of predictive healthcare analytics and Alzheimer's diagnosis. Key literature includes:

-   Arya, A.D., et al. (2023). *A systematic review on machine learning and deep learning techniques in the effective diagnosis of Alzheimer’s disease.* Brain Informatics.
-   Babulal, G.M., et al. (2021). *Alzheimer’s disease diagnosis using machine learning with clinical, imaging, and genetic data: a comparative study.* Research Square [Preprint].
-   World Health Organization. (2023). *Dementia Fact Sheet*.
-   El Kharoua, R. (2024). *Alzheimer’s Disease Dataset.* Kaggle.

---
