# Presentation

🧠 Alzheimer’s Disease

What is Alzheimer’s?
- Progressive neurodegenerative disorder that destroys memory, thinking, and daily functioning
- Most common cause of dementia (60–70% of cases)

Why Alzheimer’s is Important
- High prevalence: >55M people worldwide
- Growing burden: cases projected to triple by 2050
- Severe impact: memory loss, cognitive decline, loss of independence
- Economic cost: >$1.3 trillion annually
- Caregiver strain: emotional, physical, and financial stress

Reference:
- World Health Organization. Dementia Fact Sheet. 2023. Available at: https://www.who.int/news-room/fact-sheets/detail/dementia

-----------------------------------------------------------------

🧩 Motivation: Why Early Diagnosis Matters
- Saves lives & preserves quality of life → early interventions slow disease progression
- Reduces healthcare costs → late-stage care is far more expensive
- Improves outcomes → timely treatment, better response to therapies
- Supports families → allows emotional, financial, and medical preparation
- Drives research → earlier identification accelerates drug discovery & clinical trials

-----------------------------------------------------------------

📊 Background

Data Warehouses in Healthcare
- Integrate EHRs, lab results, imaging, and admin data in one platform
- Enable analytics, population health studies, and clinical decision support
- Increasingly important in precision medicine & real-world research

Role of Biomedical Data in Predictive Modeling
- Includes labs, genetics, imaging, and patient history
- ML models detect hidden patterns and risk factors
- Predictive analytics improves early detection, stratification, and personalized care

References
- Lyu S, Craig S, O’Reilly G, Taniar D. The development and use of data warehousing in clinical settings: a scoping review. Front Digit Health. 2025.
- Setia S, et al. Integrated Real-World Data Warehouses Across 7 Asian Health Care Systems: Scoping Review. JMIR. 2024.
- Patharkar P, Cai J, Al-Hindawi M, Wu W. Predictive Modeling of Biomedical Temporal Data in Healthcare Applications: Review and Future Directions. Front Physiol. 2024.
- Badawy M, Ramadan N, Hefny H. Healthcare predictive analytics using ML and DL: a survey. J Electr Syst Inf Technol. 2023.

-----------------------------------------------------------------

📖 Literature (1)

📖 Study Overview

📝 Title: Alzheimer’s disease diagnosis using ML with clinical, imaging, and genetic data

📍 Source: Research Square preprint (2021), DOI: 10.21203/rs.3.rs-624520/v1

🎯 Goal: Classify Alzheimer’s vs. cognitively normal (CN = healthy brain function) subjects

🗄️ Datasets:
- ADNI (Alzheimer’s Disease Neuroimaging Initiative → clinical exams + brain scans)
  → 8,320 exams, reduced to 1,000 clean samples, 22 features
- OASIS (Open Access Series of Imaging Studies → MRI scans at different dementia stages)

🔬 Features Used:
- Cognitive tests (MMSE, ADAS → measure memory and thinking ability)
- APOE4 gene (genetic risk factor for Alzheimer’s)
- MRI volumes (sizes of brain structures: hippocampus, ventricles, whole brain)
- Demographics (age, sex, education)

🛠️ Methods:
- One-hot encoding (turning categories like male/female into numbers for ML models)
- Handling missing values (filling or removing incomplete data)
- 5-fold CV (cross-validation → splitting data into 5 parts to train/test models for reliability)

📊 Results:

ADNI dataset:
- LR (Logistic Regression → simple linear model for classification) → 99.4%
- SVM (Support Vector Machine → finds boundary separating classes) → 99.1%
- RF (Random Forest → ensemble of decision trees), LDA (Linear Discriminant Analysis → finds linear combination of features to separate classes), KNN (K-Nearest Neighbors → classifies based on closest samples) → ~98%

OASIS dataset:
- LR → 84.3%
- RF → 83.9%
- SVM → 77.7%
- NB (Naive Bayes → probabilistic model assuming feature independence) lowest: 71–87%

⚠ Key point: Accuracy is very high on ADNI (may reflect overfitting → model memorizing data instead of learning patterns), more realistic on OASIS

💡 Relevance to Our Project:
- Shows strong baseline models (LR, SVM, RF) for Alzheimer’s classification
- Highlights challenge of generalization across datasets
- We differ: no hospital database access → will use open-source datasets
- Insights from this and other studies will guide our future model development and dataset selection.

📚 Reference:
- Babulal GM, Zeng H, et al. Alzheimer’s disease diagnosis using machine learning with clinical, imaging, and genetic data: a comparative study. Research Square [Preprint]. 2021. doi: 10.21203/rs.3.rs-624520/v1

-----------------------------------------------------------------

📖 Literature (2)

📖 Study Overview

📝 Title: A systematic review on machine learning and deep learning techniques in the effective diagnosis of Alzheimer’s disease

📍 Source: Brain Informatics (SpringerOpen), 2023 — Open Access  
DOI: 10.1186/s40708-023-00205-1

🎯 Goal: Summarize and compare machine learning (ML) and deep learning (DL) techniques used for accurate diagnosis and early detection of Alzheimer’s disease (AD).

🗄️ Datasets Reviewed:
- ADNI (Alzheimer’s Disease Neuroimaging Initiative) → MRI, PET, and clinical data widely used across studies.
- OASIS (Open Access Series of Imaging Studies) → MRI-based data for dementia stage classification.
- Other datasets: MIRIAD, AIBL, Kaggle-based Alzheimer datasets — smaller but diverse sources used in comparative works.

🔬 Features Commonly Used:
- Neuroimaging: MRI (structural/functional), PET (glucose metabolism, amyloid plaques)
- Clinical/Cognitive: MMSE, ADAS-Cog, demographic data (age, education)
- Genetic: APOE4 genotype
- Derived features: volumetric brain measurements (hippocampus, ventricles)

🛠️ Methods Reviewed:
- Classical ML: SVM, Random Forest, Decision Tree, Logistic Regression, KNN
- Deep Learning: 2D/3D CNNs, autoencoders, hybrid CNN–LSTM, and multimodal fusion networks
- Preprocessing techniques: normalization, skull stripping, noise filtering, augmentation
- Validation strategies: k-fold cross-validation, stratified sampling, external dataset testing

📊 Results Summary:
- ML accuracies: typically 80–95% for binary classification (AD vs CN)
- DL models (especially CNN-based) achieved >95% in some studies, but often on small or single-center datasets
- Cross-dataset validation revealed performance drops (10–20%) → highlights lack of generalization

⚠ Key Insights:
- High performance on ADNI may reflect overfitting due to data homogeneity.
- Combining multiple modalities (MRI + PET + clinical data) yields more robust results.
- Need for explainability (e.g., SHAP, Grad-CAM) and larger, balanced datasets emphasized.
- Few studies use longitudinal data, though it’s critical for predicting MCI → AD conversion.

💡 Relevance to Our Project:
- Confirms value of open datasets (ADNI, OASIS) — same sources we can access.
- Highlights effective algorithms (CNN, RF, SVM) we can benchmark against.
- Emphasizes dataset generalization and transparency, aligning with our project goals.
- Guides our preprocessing and model validation strategies for real-world applicability.

📚 Reference:
- Arya AD, et al. *A systematic review on machine learning and deep learning techniques in the effective diagnosis of Alzheimer’s disease.* Brain Informatics. 2023; 10(1): 13.  
  DOI: [10.1186/s40708-023-00205-1](https://doi.org/10.1186/s40708-023-00205-1)

-----------------------------------------------------------------

⚠ Problem Statement
- Lack of structured integration: EHRs, labs, imaging often siloed and inconsistent
- Limited interoperability → difficult to build a comprehensive patient profile
- Impact: delayed diagnosis, worse outcomes, higher costs, caregiver burden
- Need: predictive models for early detection using structured biomedical data

-----------------------------------------------------------------

💻 Technologies Used
- Data Storage & Integration: Data warehouses, ETL tools, interoperability standards (HL7, FHIR, DICOM)
- Data Processing & Analytics: Big data platforms (Spark, Hadoop), SQL/NoSQL databases
- Machine Learning & Predictive Modeling: scikit-learn, XGBoost, TensorFlow, PyTorch
- Security & Privacy: encryption, anonymization, HIPAA compliance

-----------------------------------------------------------------

🗂 Dataset Overview

Description: 
This dataset contains 2,149 patient records related to Alzheimer’s disease diagnosis and associated risk factors. It combines demographic, clinical, behavioral, and cognitive features for each subject, allowing the development of machine learning models for early detection and risk assessment.

📍 Source:  
Internal compilation (Alzheimer’s Disease Data.xlsx), inspired by open-access medical datasets such as ADNI and OASIS. The data structure mimics real-world clinical records containing both health and lifestyle parameters.

👥 Number of Patients / Records:  
- Total records: 2,149  
- Features (columns): 35  
- Target variable: `Diagnosis`  
  - 0 → Cognitively Normal (CN)  
  - 1 → Alzheimer’s Disease (AD)

🧾 Main Features:
- Demographics: Age, Gender, Ethnicity, EducationLevel  
- Lifestyle factors: Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality  
- Medical history: FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, Hypertension, HeadInjury  
- Clinical measures: SystolicBP, DiastolicBP, Cholesterol (Total, LDL, HDL, Triglycerides), BMI  
- Cognitive assessments: MMSE (Mini-Mental State Exam), FunctionalAssessment, ADL (Activities of Daily Living)  
- Behavioral symptoms: MemoryComplaints, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness  
- Administrative field: DoctorInCharge (masked for privacy)

🎯 Target Variable:  
- `Diagnosis` — represents whether the patient is diagnosed with Alzheimer’s disease (1) or is cognitively normal (0).  
This is the primary classification label used for supervised learning models in our project.

-----------------------------------------------------------------

🧹 Data Preparation & Preprocessing
- Cleaning, handling missing values, normalization, encoding categorical variables
- Can include feature selection or dimensionality reduction here

-----------------------------------------------------------------

📊 Exploratory Data Analysis (EDA)
- Visualizations, distributions, correlations, patterns in the data
- Can sometimes be combined with Data Preparation if short

-----------------------------------------------------------------

⚙ Model Selection & Building
- Choosing candidate models (e.g., Random Forest, XGBoost, SVM)
- Feature engineering and hyperparameter considerations

-----------------------------------------------------------------

🏋 Model Training & Testing
- Splitting data into train/test sets
- Performance evaluation metrics (accuracy, F1-score, ROC, etc.)

-----------------------------------------------------------------

🚀 Proposed Solution / Predictive System
- How the final model will be applied for early disease detection
- High-level system architecture or workflow

-----------------------------------------------------------------

📚 References

- World Health Organization. Dementia Fact Sheet. 2023. Available at: https://www.who.int/news-room/fact-sheets/detail/dementia
- Lyu S, Craig S, O’Reilly G, Taniar D. The development and use of data warehousing in clinical settings: a scoping review. Front Digit Health. 2025.
- Setia S, et al. Integrated Real-World Data Warehouses Across 7 Asian Health Care Systems: Scoping Review. JMIR. 2024.
- Patharkar P, Cai J, Al-Hindawi M, Wu W. Predictive Modeling of Biomedical Temporal Data in Healthcare Applications: Review and Future Directions. Front Physiol. 2024.
- Badawy M, Ramadan N, Hefny H. Healthcare predictive analytics using ML and DL: a survey. J Electr Syst Inf Technol. 2023.
- El Kharoua, R. *Alzheimer’s Disease Dataset.* Kaggle. 2024. Available at: [https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

-----------------------------------------------------------------

