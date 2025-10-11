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

## 💻 Technologies and Development Stack 
- 🐍 **Programming Language:** Python — core for data processing, ML model development, and backend logic  
- ⚙️ **Backend Framework:** FastAPI — serves ML model predictions and manages data communication  
- 🧩 **Frontend Platform:** Oracle APEX — low-code, browser-based interface for data input and visualization  
- 📁 **Data Source:** CSV files — primary storage for patient data and model results  
- 📊 **Libraries & Tools:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- 🧑‍💻 **Development Environment:** Visual Studio Code — used for coding, debugging, and project integration  

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

This work is informed by established research in predictive healthcare analytics, data warehousing, and Alzheimer's disease diagnosis. Key literature includes:

1. **World Health Organization.** (2023). *Dementia Fact Sheet.* Available at: [https://www.who.int/news-room/fact-sheets/detail/dementia](https://www.who.int/news-room/fact-sheets/detail/dementia)  
2. **Lyu, S., Craig, S., O’Reilly, G., & Taniar, D.** (2025). *The development and use of data warehousing in clinical settings: a scoping review.* *Frontiers in Digital Health.*  
3. **Setia, S., et al.** (2024). *Integrated Real-World Data Warehouses Across 7 Asian Health Care Systems: Scoping Review.* *JMIR.*  
4. **Patharkar, P., Cai, J., Al-Hindawi, M., & Wu, W.** (2024). *Predictive Modeling of Biomedical Temporal Data in Healthcare Applications: Review and Future Directions.* *Frontiers in Physiology.*  
5. **Badawy, M., Ramadan, N., & Hefny, H.** (2023). *Healthcare predictive analytics using machine learning and deep learning: a survey.* *Journal of Electrical Systems and Information Technology.*  
6. **Babulal, G.M., Zeng, H., et al.** (2021). *Alzheimer’s disease diagnosis using machine learning with clinical, imaging, and genetic data: a comparative study.* *Research Square* [Preprint]. DOI: [10.21203/rs.3.rs-624520/v1](https://doi.org/10.21203/rs.3.rs-624520/v1)  
7. **Arya, A.D., et al.** (2023). *A systematic review on machine learning and deep learning techniques in the effective diagnosis of Alzheimer’s disease.* *Brain Informatics*, 10(1), 13. DOI: [10.1186/s40708-023-00205-1](https://doi.org/10.1186/s40708-023-00205-1)  
8. **El Kharoua, R.** (2024). *Alzheimer’s Disease Dataset.* *Kaggle.* Available at: [https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)  
9. **Cochrane, C., Castineira, D., Shiban, N., & Protopapas, P.** (2020). *Application of Machine Learning to Predict the Risk of Alzheimer's Disease: An Accurate and Practical Solution for Early Diagnostics.* *arXiv preprint* [abs/2006.08702](https://arxiv.org/abs/2006.08702)  
