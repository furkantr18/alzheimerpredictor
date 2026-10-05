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

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Installation Steps

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

**2. Install the required dependencies:**

The project requires several Python packages listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- pandas, numpy (data processing)
- scikit-learn (machine learning models)
- xgboost, lightgbm, catboost (gradient boosting models)
- matplotlib, seaborn (visualization)
- shap, lime (model explainability)
- fastapi (API deployment)

**3. Run the model training pipeline:**

To train all models and generate evaluation reports:

```bash
python src/model\ training/model_trainer.py
```

Or navigate to the directory:
```bash
cd "src/model training"
python model_trainer.py
```

### MRI Image Preprocessing (For Dementia Class Classification)

If you added NEW, UNPROCESSED MRI images organized by class folders under `src/data/img_unprocessed/` (for example `MildDemented/`, `ModerateDemented/`, `NonDemented/`, `VeryMildDemented/`), run:

```bash
python src/imgProcessing/preprocess_mri.py --input-dir src/data/img_unprocessed
```

This pipeline performs MRI-focused preprocessing including:
- brain-region masking and ROI cropping
- denoising (Gaussian, median, bilateral, Wiener)
- contrast enhancement (CLAHE, optional histogram equalization)
- intensity normalization and discretization (binning)
- feature enhancement (unsharp + edge enhancement)
- geometric standardization (resize)
- augmentation (rotation, scale, shear, flip, noise, blur, brightness/contrast)

Outputs are written to:
- `src/data/processed/mri/clean/<class>/`
- `src/data/processed/mri/augmented/<class>/`
- `src/data/processed/mri/manifest.csv`

Example with custom settings:

```bash
python src/imgProcessing/preprocess_mri.py --target-size 224 --augment-per-image 2 --rotation-correction
```

### MRI Model Training (Image Classification)

Train multiple MRI classifiers (Logistic Regression, SVM, Random Forest, KNN, MLP, and XGBoost if installed), compare results, and save the best model:

```bash
python src/imgProcessing/mri_model_trainer.py --include-augmented
```

Training artifacts are written to:
- `src/imgProcessing/output/model_comparison.csv`
- `src/imgProcessing/output/models/best_mri_model.pkl`
- `src/imgProcessing/output/classification_report.txt`
- `src/imgProcessing/output/plots/best_model_confusion_matrix.png`
- `src/imgProcessing/output/training_summary.json`

### MRI Deep Learning Training (Transfer Learning)

Train transfer learning MRI classifiers with pretrained backbones:
- `resnet50`
- `efficientnet_b0`

Examples:

```bash
python src/imgProcessing/mri_dl_trainer.py --architecture resnet50 --epochs 20
python src/imgProcessing/mri_dl_trainer.py --architecture efficientnet_b0 --epochs 25 --batch-size 16
```

Deep learning artifacts are written to:
- `src/imgProcessing/output/models/best_mri_dl_model.pt`
- `src/imgProcessing/output/models/best_mri_dl_model_metadata.json`
- `src/imgProcessing/output/dl_training_history.csv`
- `src/imgProcessing/output/dl_classification_report.txt`
- `src/imgProcessing/output/plots/best_mri_dl_confusion_matrix.png`
- `src/imgProcessing/output/dl_training_summary.json`

### MRI Inference

Single image prediction:

```bash
python src/imgProcessing/predict_mri.py --image path/to/image.jpg
```

Batch prediction from directory:

```bash
python src/imgProcessing/predict_mri.py --image-dir path/to/folder
```

### MRI Deep Learning Inference

Single image prediction:

```bash
python src/imgProcessing/predict_mri_dl.py --image path/to/image.jpg
```

Batch directory prediction:

```bash
python src/imgProcessing/predict_mri_dl.py --image-dir path/to/folder
```

**4. Launch Jupyter Notebook (optional):**

For interactive data exploration:
```bash
jupyter notebook
```
Open the main notebook file to explore the analysis and models.

## 🔮 Making Predictions

After training the models, you can use the prediction script to classify new patient data and get risk assessments.

### Interactive Mode (Manual Input)

Enter patient data manually and get instant predictions:

```bash
python src/predict.py --interactive
```

You'll be prompted to enter values for each feature, with helpful ranges and averages displayed.

### Single Patient Prediction

Predict for a single patient from a CSV file:

```bash
python src/predict.py --patient path/to/patient_data.csv
```

### Batch Prediction (Multiple Patients)

Process multiple patients at once and save results:

```bash
python src/predict.py --batch path/to/patients.csv --output predictions.csv
```

### Use Specific Model

By default, the best-performing model is used. To use a specific model:

```bash
python src/predict.py --interactive --model "src/model training/output/trained_models/xgboost.pkl"
```

### Prediction Output

The prediction system provides comprehensive results:

- **🔍 Diagnosis**: Cognitive Normal (CN) or Alzheimer's Disease (AD)
- **📊 Confidence Score**: Probability percentage (0-100%)
- **📈 Probability Breakdown**: Visual bars showing probabilities for each class
- **💡 Risk Interpretation**: HIGH/MODERATE/LOW risk assessment with medical recommendations

**Example Output:**

```
======================================================================
PREDICTION RESULT
======================================================================

🔍 Diagnosis: Alzheimer's Disease
📊 Confidence Score: 87.32%

📈 Probability Breakdown:
  Cognitive Normal               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12.68%
  Alzheimer's Disease            ████████████████████████████████████░░░░ 87.32%

💡 Interpretation:
  ⚠️  HIGH RISK: Strong indicators of Alzheimer's disease detected.
     Recommendation: Consult with a neurologist immediately.

======================================================================
⚕️  DISCLAIMER: This is a predictive model and NOT a medical diagnosis.
   Always consult with qualified healthcare professionals.
======================================================================
```

### Input Data Format

For batch predictions, your CSV file should contain the same features as the training data:

```csv
PatientID,Age,Gender,Ethnicity,EducationLevel,BMI,Smoking,AlcoholConsumption,...
P001,0.65,1,0,0.67,0.42,0,0.5,...
P002,0.82,0,1,0.33,0.38,1,0.3,...
```

⚕️ **IMPORTANT DISCLAIMER**: This prediction tool is for research and educational purposes. It provides risk assessments based on machine learning models and should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## 🤖 Model Selection & Feature Engineering

### Machine Learning Models

We employ a comprehensive ensemble of models to capture different aspects of the data:

#### **Traditional Models**
- **Logistic Regression (LR):** Simple, interpretable baseline for binary classification
- **Support Vector Machine (SVM):** Effective for high-dimensional features with kernel tricks
- **K-Nearest Neighbors (KNN):** Intuitive, instance-based model for pattern recognition
- **Gaussian Naive Bayes:** Probabilistic classifier based on Bayes' theorem

#### **Tree-Based Ensemble Models**
- **Decision Trees:** Interpretable decision rules for classification
- **Random Forest (RF):** Ensemble of decision trees to capture non-linear relationships
- **XGBoost:** Gradient boosting with regularization for strong tabular performance
- **LightGBM:** Fast gradient boosting with leaf-wise tree growth
- **CatBoost:** Gradient boosting optimized for categorical features

#### **Advanced Ensemble**
- **Stacking Ensemble:** Meta-learner combining predictions from multiple base models

### Feature Engineering Pipeline

Our feature engineering process includes:

1. **Feature Selection:** 
   - Identify and select relevant features based on domain knowledge and statistical tests
   - Remove non-predictive identifiers (e.g., PatientID)

2. **Data Quality:**
   - Handle missing values using appropriate imputation strategies
   - Detect and handle outliers using IQR-based methods
   
3. **Feature Transformation:**
   - Encode categorical variables (Label/One-Hot encoding)
   - Normalize/scale numerical features (StandardScaler/RobustScaler)

4. **Feature Importance:**
   - Analyze feature contributions using SHAP and LIME
   - Visualize model interpretability and decision-making

### Hyperparameter Tuning

We optimize model performance through systematic hyperparameter search:

- **Random Forest:** Number of trees, max depth, min samples split
- **XGBoost/LightGBM/CatBoost:** Learning rate, max depth, subsample ratio, regularization
- **SVM:** Kernel type, C (regularization), gamma
- **Search Strategy:** Randomized search with cross-validation

### Model Evaluation

#### Cross-Validation Strategy
- **5-Fold Stratified Cross-Validation** to ensure robust performance estimates
- Prevents overfitting and validates generalization to unseen data

#### Performance Metrics
- **Accuracy:** Overall classification correctness
- **F1-Score:** Harmonic mean of precision and recall (weighted for multiclass)
- **ROC-AUC:** Area under the receiver operating characteristic curve
- **Confusion Matrix:** Detailed breakdown of predictions vs. actual labels

#### Model Explainability
- **SHAP (SHapley Additive exPlanations):** Global feature importance
- **LIME (Local Interpretable Model-agnostic Explanations):** Instance-level predictions

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
