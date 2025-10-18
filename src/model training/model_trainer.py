"""
Alzheimer's Disease Prediction - Model Training Pipeline
========================================================
This module implements a comprehensive machine learning pipeline for predicting
Alzheimer's disease diagnosis using various classification algorithms.

Features:
- Multiple classification models (Logistic Regression, SVM, KNN, Naive Bayes, 
  Decision Trees, XGBoost, LightGBM, CatBoost, Ensemble Stacking)
- Feature engineering and preprocessing
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- 5-fold cross-validation
- Comprehensive evaluation metrics (Accuracy, F1-score, ROC-AUC)
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Set matplotlib backend before importing pyplot (fixes Windows tkinter issues)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier

# Gradient boosting imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Metrics imports
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc, make_scorer
)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Explainability imports
import shap
from lime.lime_tabular import LimeTabularExplainer

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_PATH, BEST_MODEL_PATH

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TeeOutput:
    """
    A utility class to duplicate print output to both console and buffer for later saving.
    
    This allows all terminal output to be saved to a text file at the end while still
    displaying in the console during execution.
    """
    
    def __init__(self, log_file_path):
        """
        Initialize the TeeOutput.
        
        Parameters:
        -----------
        log_file_path : str or Path
            Path to the log file where output will be written at the end
        """
        self.terminal = sys.stdout
        self.log_file_path = log_file_path
        self.buffer = []  # Store all output in memory
    
    def write(self, message):
        """Write message to terminal and buffer."""
        self.terminal.write(message)
        self.buffer.append(message)  # Store in memory
    
    def flush(self):
        """Flush terminal."""
        self.terminal.flush()
    
    def save_to_file(self):
        """Save buffered output to file at the end."""
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(self.buffer))
    
    def close(self):
        """Save buffer to file and clean up."""
        self.save_to_file()


class AlzheimerModelTrainer:
    """
    A comprehensive model training pipeline for Alzheimer's disease prediction.
    
    This class handles:
    - Data loading and preprocessing
    - Feature engineering
    - Missing value imputation
    - Outlier detection and handling
    - Feature scaling and normalization
    - Model training with multiple algorithms
    - Hyperparameter tuning
    - Model evaluation with cross-validation
    - Results visualization and reporting
    """
    
    def __init__(self, data_path, target_column='Diagnosis', random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        data_path : str
            Path to the processed data CSV file
        target_column : str
            Name of the target column (default: 'Diagnosis')
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        
        # Initialize data containers
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize preprocessing objects
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
        # Initialize results storage
        self.trained_models = {}
        self.model_results = {}
        self.best_models = {}
        
        # Create output directories
        self.output_dir = Path('src/model training/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file for output
        self.log_file_path = self.output_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self.tee_output = None
        
        print(f"[INFO] AlzheimerModelTrainer initialized with random_state={random_state}")
    
    
    def load_data(self):
        """
        Load the dataset from the specified path.
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        print("\n" + "="*70)
        print("[STEP 1] Loading Data")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[SUCCESS] Data loaded successfully!")
            print(f"[INFO] Dataset shape: {self.df.shape}")
            print(f"[INFO] Columns: {list(self.df.columns)}")
            
            # Display basic information
            print(f"\n[INFO] Target variable distribution:")
            print(self.df[self.target_column].value_counts())
            
            # === Clustering and Correlation Visualizations ===
            plots_root = self.output_dir / "plots"
            plots_root.mkdir(exist_ok=True)

            # 1. Hierarchical Clustering Dendrogram (Ward linkage)
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
            # Select only numeric columns and exclude non-predictive ID columns
            X_num = self.df.select_dtypes(include=[np.number]).drop(columns=[self.target_column], errors='ignore')
            id_columns = ['PatientID', 'DoctorInCharge']
            for col in id_columns:
                if col in X_num.columns:
                    X_num = X_num.drop(columns=[col])
            Z = linkage(X_num, method='ward')
            
            # Get cluster information for interpretation
            # Cut tree at a distance that creates a reasonable number of clusters
            max_distance = Z[-1, 2]  # Maximum distance in the linkage
            threshold = max_distance * 0.5  # Cut at 50% of max distance
            clusters = fcluster(Z, threshold, criterion='distance')
            n_clusters = len(np.unique(clusters))
            
            # Get cluster sizes
            unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
            
            plt.figure(figsize=(20, 10))
            # Don't show individual labels for large datasets - just show clusters
            dendro = dendrogram(Z, no_labels=True, color_threshold=threshold)
            plt.title(f'Hierarchical Clustering Dendrogram (Ward linkage)\n{n_clusters} clusters identified at distance threshold = {threshold:.2f}', 
                     fontsize=16, pad=20)
            plt.xlabel('Samples (Each color represents a distinct cluster group)', fontsize=14)
            plt.ylabel('Ward Distance (Lower = More Similar)', fontsize=14)
            
            # Add horizontal line showing the threshold
            plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, 
                       label=f'Cluster threshold ({n_clusters} clusters)')
            
            # Add text annotations for cluster numbers and sizes
            cluster_info_text = "Cluster Sizes: " + ", ".join([f"C{i+1}: {count}" for i, count in enumerate(cluster_counts)])
            plt.text(0.5, 0.98, cluster_info_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            dendro_path = plots_root / 'hierarchical_dendrogram.png'
            plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Dendrogram saved to: {dendro_path}")
            print(f"[INFO] Identified {n_clusters} hierarchical clusters at threshold {threshold:.2f}")
            print(f"[INFO] Cluster sizes: {dict(zip([f'Cluster {i+1}' for i in range(len(cluster_counts))], cluster_counts))}")

            # 2. Elbow Method for KMeans
            from sklearn.cluster import KMeans
            sse = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_num)
                sse.append(kmeans.inertia_)
            plt.figure(figsize=(8, 5))
            plt.plot(K_range, sse, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method for Optimal k (KMeans)')
            plt.tight_layout()
            elbow_path = plots_root / 'elbow_method.png'
            plt.savefig(elbow_path, dpi=200)
            plt.close()
            print(f"[INFO] Elbow method plot saved to: {elbow_path}")

            # 3. Silhouette Score Plot
            from sklearn.metrics import silhouette_score
            sil_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_num)
                sil = silhouette_score(X_num, labels)
                sil_scores.append(sil)
            plt.figure(figsize=(8, 5))
            plt.plot(range(2, 11), sil_scores, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score for KMeans Clustering')
            plt.tight_layout()
            sil_path = plots_root / 'silhouette_scores.png'
            plt.savefig(sil_path, dpi=200)
            plt.close()
            print(f"[INFO] Silhouette score plot saved to: {sil_path}")

            # 4. Strongest Feature Correlations Plot (top 10 pairs)
            corr = X_num.corr().abs()
            pairs = corr.unstack().sort_values(ascending=False)
            pairs = pairs[pairs < 1].drop_duplicates().head(10)
            plt.figure(figsize=(10, 6))
            pairs.plot(kind='bar')
            plt.title('Top 10 Strongest Feature Correlations')
            plt.ylabel('Correlation Coefficient')
            plt.tight_layout()
            strong_corr_path = plots_root / 'strongest_feature_correlations.png'
            plt.savefig(strong_corr_path, dpi=200)
            plt.close()
            print(f"[INFO] Strongest feature correlations plot saved to: {strong_corr_path}")

            # 5. Feature Correlation Heatmap
            plt.figure(figsize=(24, 20))
            sns.heatmap(X_num.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, 
                       annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Heatmap', fontsize=18, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            heatmap_path = plots_root / 'feature_correlation_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Feature correlation heatmap saved to: {heatmap_path}")

            return self.df
        
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            raise
    
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Numerical columns: Impute with median (robust to outliers)
        - Categorical columns: Impute with mode (most frequent value)
        """
        print("\n" + "="*70)
        print("[STEP 2] Handling Missing Values")
        print("="*70)
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        
        if len(missing_columns) == 0:
            print("[INFO] No missing values found in the dataset.")
            return
        
        print(f"[INFO] Found missing values in {len(missing_columns)} columns:")
        print(missing_columns)
        
        # Separate numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from the lists if present
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        # Impute numerical columns with median
        if any(col in numerical_cols for col in missing_columns.index):
            num_imputer = SimpleImputer(strategy='median')
            self.df[numerical_cols] = num_imputer.fit_transform(self.df[numerical_cols])
            print("[SUCCESS] Numerical columns imputed with median values.")
        
        # Impute categorical columns with mode
        if any(col in categorical_cols for col in missing_columns.index):
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])
            print("[SUCCESS] Categorical columns imputed with mode values.")
        
        print(f"[INFO] Missing values after imputation: {self.df.isnull().sum().sum()}")
    
    
    def detect_and_handle_outliers(self, method='iqr', threshold=1.5):
        """
        Detect and handle outliers in numerical features.
        
        Parameters:
        -----------
        method : str
            Method for outlier detection ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection (default: 1.5 for IQR method)
        """
        print("\n" + "="*70)
        print("[STEP 3] Detecting and Handling Outliers")
        print("="*70)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        
        outlier_summary = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                # IQR method (Interquartile Range)
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Count outliers
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers at bounds instead of removing
                    self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
                    outlier_summary[col] = outliers
        
        if outlier_summary:
            print(f"[INFO] Outliers detected and capped in {len(outlier_summary)} columns:")
            for col, count in outlier_summary.items():
                print(f"  - {col}: {count} outliers")
        else:
            print("[INFO] No significant outliers detected.")
        
        print(f"[SUCCESS] Outlier handling completed.")
    
    
    def encode_categorical_features(self):
        """
        Encode categorical features using Label Encoding.
        
        For ordinal categories, maintains order.
        For nominal categories, assigns integer labels.
        """
        print("\n" + "="*70)
        print("[STEP 4] Encoding Categorical Features")
        print("="*70)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's categorical (will be encoded separately)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        if len(categorical_cols) == 0:
            print("[INFO] No categorical features to encode.")
            return
        
        print(f"[INFO] Encoding {len(categorical_cols)} categorical columns:")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  - {col}: {len(le.classes_)} unique categories")
        
        print("[SUCCESS] Categorical encoding completed.")
    
    
    def feature_selection(self, method='mutual_info', k_features=None):
        """
        Select the most relevant features using statistical methods.
        
        Parameters:
        -----------
        method : str
            Feature selection method ('mutual_info' or 'f_classif')
        k_features : int or None
            Number of top features to select (None = keep all features)
        """
        print("\n" + "="*70)
        print("[STEP 5] Feature Selection")
        print("="*70)
        
        if k_features is None:
            print("[INFO] Skipping feature selection (keeping all features).")
            return
        
        # Ensure we don't select more features than available
        k_features = min(k_features, self.X.shape[1])
        
        print(f"[INFO] Selecting top {k_features} features using {method} method...")
        
        # Choose scoring function
        if method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        # Perform feature selection
        selector = SelectKBest(score_func=score_func, k=k_features)
        X_selected = selector.fit_transform(self.X, self.y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = self.feature_names[selected_mask]
        
        print(f"[INFO] Selected features: {list(selected_features)}")
        
        # Update X with selected features
        self.X = pd.DataFrame(X_selected, columns=selected_features)
        self.feature_names = selected_features
        
        print(f"[SUCCESS] Feature selection completed. New shape: {self.X.shape}")
    
    
    def prepare_features_and_target(self):
        """
        Prepare features (X) and target (y) for model training.
        
        Separates the dataset into feature matrix and target vector.
        """
        print("\n" + "="*70)
        print("[STEP 6] Preparing Features and Target")
        print("="*70)
        
        # Encode target variable if it's categorical
        if self.df[self.target_column].dtype == 'object':
            le_target = LabelEncoder()
            self.y = le_target.fit_transform(self.df[self.target_column])
            self.label_encoders['target'] = le_target
            print(f"[INFO] Target variable encoded: {dict(enumerate(le_target.classes_))}")
        else:
            self.y = self.df[self.target_column].values
        
        # Prepare features - drop target column and non-predictive ID columns
        columns_to_drop = [self.target_column]
        # Drop ID columns that don't have predictive value
        id_columns = ['PatientID', 'DoctorInCharge']
        for col in id_columns:
            if col in self.df.columns:
                columns_to_drop.append(col)
                print(f"[INFO] Dropping non-predictive ID column: {col}")
        
        self.X = self.df.drop(columns=columns_to_drop)
        self.feature_names = self.X.columns.values
        
        print(f"[INFO] Features shape: {self.X.shape}")
        print(f"[INFO] Target shape: {self.y.shape}")
        print(f"[INFO] Target classes: {np.unique(self.y)}")
        print("[SUCCESS] Features and target prepared.")
    
    
    def split_data(self, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in test split (default: 0.2)
        stratify : bool
            Whether to stratify the split by target variable (default: True)
        """
        print("\n" + "="*70)
        print("[STEP 7] Splitting Data")
        print("="*70)
        
        stratify_arg = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_arg
        )
        
        print(f"[INFO] Training set size: {self.X_train.shape[0]} samples")
        print(f"[INFO] Testing set size: {self.X_test.shape[0]} samples")
        print(f"[INFO] Training set class distribution: {np.bincount(self.y_train)}")
        print(f"[INFO] Testing set class distribution: {np.bincount(self.y_test)}")
        print("[SUCCESS] Data split completed.")
    
    
    def scale_features(self, method='standard'):
        """
        Scale/normalize features for better model performance.
        
        Parameters:
        -----------
        method : str
            Scaling method ('standard' for StandardScaler, 'robust' for RobustScaler)
        """
        print("\n" + "="*70)
        print("[STEP 8] Scaling Features")
        print("="*70)
        
        # Choose scaler
        if method == 'robust':
            self.scaler = RobustScaler()
            print("[INFO] Using RobustScaler (robust to outliers)...")
        else:
            self.scaler = StandardScaler()
            print("[INFO] Using StandardScaler (zero mean, unit variance)...")
        
        # Fit on training data and transform both train and test
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Save scaler/preprocessor object for deployment
        import joblib
        trained_models_dir = self.output_dir / "trained_models"
        trained_models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = trained_models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"[INFO] Scaler/preprocessor saved to: {scaler_path}")

        print(f"[SUCCESS] Feature scaling completed.")
    
    
    def train_logistic_regression(self, tune_hyperparameters=True):
        """
        Train Logistic Regression model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Logistic Regression")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
                'penalty': ['l1', 'l2'],  # Regularization type
                'solver': ['liblinear', 'saga'],  # Optimization algorithm
                'max_iter': [1000]
            }
            
            # Grid search with 5-fold cross-validation
            model = LogisticRegression(random_state=self.random_state)
            grid_search = GridSearchCV(
                model, param_grid,
                cv=5, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            print(f"[INFO] Best parameters: {grid_search.best_params_}")
            print(f"[INFO] Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['Logistic Regression'] = best_model
        
        # Evaluate the model
        self._evaluate_model('Logistic Regression', best_model)
        
        print("[SUCCESS] Logistic Regression training completed.")
    
    
    def train_svm(self, tune_hyperparameters=True):
        """
        Train Support Vector Machine (SVM) model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Support Vector Machine (SVM)")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],  # Regularization parameter
                'kernel': ['rbf', 'linear', 'poly'],  # Kernel type
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
                'probability': [True]  # Enable probability estimates for ROC-AUC
            }
            
            # Randomized search (faster than grid search for SVM)
            model = SVC(random_state=self.random_state)
            random_search = RandomizedSearchCV(
                model, param_grid,
                n_iter=20, cv=5,
                scoring='f1_weighted',
                n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            
            print(f"[INFO] Best parameters: {random_search.best_params_}")
            print(f"[INFO] Best CV score: {random_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = SVC(random_state=self.random_state, probability=True)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['SVM'] = best_model
        
        # Evaluate the model
        self._evaluate_model('SVM', best_model)
        
        print("[SUCCESS] SVM training completed.")
    
    
    def train_knn(self, tune_hyperparameters=True):
        """
        Train K-Nearest Neighbors (KNN) model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training K-Nearest Neighbors (KNN)")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 20],  # Number of neighbors
                'weights': ['uniform', 'distance'],  # Weight function
                'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
                'p': [1, 2]  # Power parameter for Minkowski metric
            }
            
            # Grid search with 5-fold cross-validation
            model = KNeighborsClassifier()
            grid_search = GridSearchCV(
                model, param_grid,
                cv=5, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            print(f"[INFO] Best parameters: {grid_search.best_params_}")
            print(f"[INFO] Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = KNeighborsClassifier()
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['KNN'] = best_model
        
        # Evaluate the model
        self._evaluate_model('KNN', best_model)
        
        print("[SUCCESS] KNN training completed.")
    
    
    def train_naive_bayes(self):
        """
        Train Gaussian Naive Bayes model.
        
        Note: Naive Bayes has minimal hyperparameters to tune.
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Gaussian Naive Bayes")
        print("-"*70)
        
        # Train Gaussian Naive Bayes (minimal hyperparameters)
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['Naive Bayes'] = model
        
        # Evaluate the model
        self._evaluate_model('Naive Bayes', model)
        
        print("[SUCCESS] Naive Bayes training completed.")
    
    
    def train_decision_tree(self, tune_hyperparameters=True):
        """
        Train Decision Tree model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Decision Tree")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'max_depth': [3, 5, 7, 10, 15, None],  # Maximum tree depth
                'min_samples_split': [2, 5, 10, 20],  # Min samples to split a node
                'min_samples_leaf': [1, 2, 4, 8],  # Min samples at leaf node
                'criterion': ['gini', 'entropy'],  # Split quality measure
                'splitter': ['best', 'random']  # Split strategy
            }
            
            # Grid search with 5-fold cross-validation
            model = DecisionTreeClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                model, param_grid,
                cv=5, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            print(f"[INFO] Best parameters: {grid_search.best_params_}")
            print(f"[INFO] Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = DecisionTreeClassifier(random_state=self.random_state)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['Decision Tree'] = best_model
        
        # Evaluate the model
        self._evaluate_model('Decision Tree', best_model)
        
        print("[SUCCESS] Decision Tree training completed.")
    
    
    def train_random_forest(self, tune_hyperparameters=True):
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Random Forest")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200, 300],  # Number of trees
                'max_depth': [5, 10, 15, 20, None],  # Maximum tree depth
                'min_samples_split': [2, 5, 10],  # Min samples to split
                'min_samples_leaf': [1, 2, 4],  # Min samples at leaf
                'max_features': ['sqrt', 'log2', None],  # Features to consider for split
                'bootstrap': [True, False]  # Bootstrap samples
            }
            
            # Randomized search (faster than grid search)
            model = RandomForestClassifier(random_state=self.random_state)
            random_search = RandomizedSearchCV(
                model, param_grid,
                n_iter=30, cv=5,
                scoring='f1_weighted',
                n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            
            print(f"[INFO] Best parameters: {random_search.best_params_}")
            print(f"[INFO] Best CV score: {random_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['Random Forest'] = best_model
        
        # Evaluate the model
        self._evaluate_model('Random Forest', best_model)
        
        print("[SUCCESS] Random Forest training completed.")
    
    
    def train_xgboost(self, tune_hyperparameters=True):
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training XGBoost")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],  # Number of boosting rounds
                'max_depth': [3, 5, 7, 9],  # Maximum tree depth
                'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Boosting learning rate
                'subsample': [0.6, 0.8, 1.0],  # Subsample ratio
                'colsample_bytree': [0.6, 0.8, 1.0],  # Feature subsample ratio
                'gamma': [0, 0.1, 0.5, 1],  # Minimum loss reduction
                'reg_alpha': [0, 0.1, 1],  # L1 regularization
                'reg_lambda': [0, 0.1, 1]  # L2 regularization
            }
            
            # Randomized search
            model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            random_search = RandomizedSearchCV(
                model, param_grid,
                n_iter=30, cv=5,
                scoring='f1_weighted',
                n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            
            print(f"[INFO] Best parameters: {random_search.best_params_}")
            print(f"[INFO] Best CV score: {random_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['XGBoost'] = best_model
        
        # Evaluate the model
        self._evaluate_model('XGBoost', best_model)
        
        print("[SUCCESS] XGBoost training completed.")
    
    
    def train_lightgbm(self, tune_hyperparameters=True):
        """
        Train LightGBM model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training LightGBM")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],  # Number of boosting rounds
                'max_depth': [3, 5, 7, 9, -1],  # Maximum tree depth (-1 = no limit)
                'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Boosting learning rate
                'num_leaves': [15, 31, 63, 127],  # Maximum number of leaves
                'subsample': [0.6, 0.8, 1.0],  # Subsample ratio
                'colsample_bytree': [0.6, 0.8, 1.0],  # Feature subsample ratio
                'reg_alpha': [0, 0.1, 1],  # L1 regularization
                'reg_lambda': [0, 0.1, 1]  # L2 regularization
            }
            
            # Randomized search
            model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            random_search = RandomizedSearchCV(
                model, param_grid,
                n_iter=30, cv=5,
                scoring='f1_weighted',
                n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            
            print(f"[INFO] Best parameters: {random_search.best_params_}")
            print(f"[INFO] Best CV score: {random_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = LGBMClassifier(random_state=self.random_state, verbose=-1)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['LightGBM'] = best_model
        
        # Evaluate the model
        self._evaluate_model('LightGBM', best_model)
        
        print("[SUCCESS] LightGBM training completed.")
    
    
    def train_catboost(self, tune_hyperparameters=True):
        """
        Train CatBoost model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (default: True)
        """
        print("\n" + "-"*70)
        print("[MODEL] Training CatBoost")
        print("-"*70)
        
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'iterations': [100, 200, 300],  # Number of boosting rounds
                'depth': [4, 6, 8, 10],  # Maximum tree depth
                'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate
                'l2_leaf_reg': [1, 3, 5, 7],  # L2 regularization
                'border_count': [32, 64, 128],  # Number of splits for numerical features
                'bagging_temperature': [0, 1, 5]  # Bayesian bootstrap parameter
            }
            
            # Randomized search
            model = CatBoostClassifier(random_state=self.random_state, verbose=0)
            random_search = RandomizedSearchCV(
                model, param_grid,
                n_iter=20, cv=5,
                scoring='f1_weighted',
                n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            
            print(f"[INFO] Best parameters: {random_search.best_params_}")
            print(f"[INFO] Best CV score: {random_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = CatBoostClassifier(random_state=self.random_state, verbose=0)
            best_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['CatBoost'] = best_model
        
        # Evaluate the model
        self._evaluate_model('CatBoost', best_model)
        
        print("[SUCCESS] CatBoost training completed.")
    
    
    def train_stacking_ensemble(self):
        """
        Train a Stacking Ensemble model that combines multiple base models.
        
        Uses the best performing models as base estimators and Logistic Regression
        as the meta-classifier (final estimator).
        """
        print("\n" + "-"*70)
        print("[MODEL] Training Stacking Ensemble")
        print("-"*70)
        
        # Define base estimators (use previously trained models or create new ones)
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('xgb', XGBClassifier(random_state=self.random_state, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(random_state=self.random_state, verbose=-1)),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ]
        
        # Define meta-classifier (final estimator)
        meta_classifier = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_classifier,
            cv=5,  # 5-fold cross-validation for base models
            n_jobs=-1
        )
        
        print("[INFO] Training stacking ensemble with base models: Random Forest, XGBoost, LightGBM, SVM")
        print("[INFO] Meta-classifier: Logistic Regression")
        
        # Train the stacking model
        stacking_model.fit(self.X_train, self.y_train)
        
        # Store the trained model
        self.trained_models['Stacking Ensemble'] = stacking_model
        
        # Evaluate the model
        self._evaluate_model('Stacking Ensemble', stacking_model)
        
        print("[SUCCESS] Stacking Ensemble training completed.")
    
    
    def _evaluate_model(self, model_name, model):
        """
        Evaluate a trained model using multiple metrics and cross-validation.
        
        Parameters:
        -----------
        model_name : str
            Name of the model for reporting
        model : estimator
            Trained sklearn-compatible model
        """
        print(f"\n[EVALUATION] Evaluating {model_name}...")
        
        # Make predictions on test set
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics on test set
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # ROC-AUC calculation (requires probability predictions)
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                # For binary classification
                if len(np.unique(self.y)) == 2:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                # For multi-class classification
                else:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
        except Exception as e:
            print(f"[WARNING] Could not calculate ROC-AUC: {str(e)}")
            roc_auc = None
        
        # Perform 5-fold cross-validation
        cv_scores_accuracy = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_scores_f1 = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
        
        # Store results
        self.model_results[model_name] = {
            'test_accuracy': accuracy,
            'test_f1_score': f1,
            'test_roc_auc': roc_auc,
            'cv_accuracy_mean': cv_scores_accuracy.mean(),
            'cv_accuracy_std': cv_scores_accuracy.std(),
            'cv_f1_mean': cv_scores_f1.mean(),
            'cv_f1_std': cv_scores_f1.std(),
            'predictions': y_pred
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"{model_name} - Test Set Performance:")
        print(f"{'='*50}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"\n{model_name} - 5-Fold Cross-Validation:")
        print(f"{'='*50}")
        print(f"  Accuracy:  {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
        print(f"  F1-Score:  {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std():.4f})")
        print(f"{'='*50}\n")

        # Create a central 'plots' directory and subdirectory for this model
        plots_root = self.output_dir / "plots"
        plots_root.mkdir(exist_ok=True)
        model_plot_dir = plots_root / f"{model_name.replace(' ', '_').lower()}"
        model_plot_dir.mkdir(exist_ok=True)

        # Confusion Matrix Plot
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = model_plot_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

        # ROC Curve Plot (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                n_classes = len(np.unique(self.y_test))
                plt.figure(figsize=(7, 6))
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] ROC curve plot saved to: {roc_path}")
                else:
                    # Multiclass ROC curve
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(self.y_test == i, y_pred_proba[:, i])
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Multiclass ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] Multiclass ROC curve plot saved to: {roc_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot ROC curve: {str(e)}")

        # SHAP Feature Importance (global)
        try:
            explainer = None
            shap_values = None
            X_test_shap = self.X_test  # Full test set by default
            
            # Tree-based models
            if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_test)
            # Linear models
            elif hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, self.X_train, feature_perturbation="interventional")
                shap_values = explainer.shap_values(self.X_test)
            # Kernel for others (KNN, Naive Bayes, etc.)
            else:
                # Use subset for KernelExplainer (it's slow)
                X_test_shap = self.X_test[:50]
                explainer = shap.KernelExplainer(model.predict, self.X_train)
                shap_values = explainer.shap_values(X_test_shap, nsamples=100)

            # Create SHAP plot with proper sizing and labels
            plt.figure(figsize=(12, 8))
            # Handle multiclass case where shap_values is a list
            if isinstance(shap_values, list):
                # For multiclass, plot the first class or average across classes
                shap.summary_plot(shap_values[0], X_test_shap, feature_names=self.feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_test_shap, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            shap_path = model_plot_dir / 'shap_summary.png'
            plt.savefig(shap_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"[INFO] SHAP summary plot saved to: {shap_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate SHAP plot: {str(e)}")

        # LIME Local Explanations (instance-level)
        try:
            lime_explainer = LimeTabularExplainer(
                training_data=np.array(self.X_train),
                feature_names=self.feature_names,
                class_names=[str(c) for c in np.unique(self.y)],
                mode='classification',
                discretize_continuous=True
            )
            # Explain first 3 test samples
            for i in range(min(3, len(self.X_test))):
                exp = lime_explainer.explain_instance(
                    np.array(self.X_test[i]),
                    model.predict_proba,
                    num_features=10
                )
                lime_path = model_plot_dir / f'lime_explanation_{i+1}.png'
                exp.as_pyplot_figure()
                plt.tight_layout()
                plt.savefig(lime_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"[INFO] LIME explanation plot saved to: {lime_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate LIME explanations: {str(e)}")

        # Confusion Matrix Plot
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = model_plot_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

        # ROC Curve Plot (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                n_classes = len(np.unique(self.y_test))
                plt.figure(figsize=(7, 6))
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] ROC curve plot saved to: {roc_path}")
                else:
                    # Multiclass ROC curve
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(self.y_test == i, y_pred_proba[:, i])
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Multiclass ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] Multiclass ROC curve plot saved to: {roc_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot ROC curve: {str(e)}")


        # Create a central 'plots' directory and subdirectory for this model
        plots_root = self.output_dir / "plots"
        plots_root.mkdir(exist_ok=True)
        model_plot_dir = plots_root / f"{model_name.replace(' ', '_').lower()}"
        model_plot_dir.mkdir(exist_ok=True)

        # Confusion Matrix Plot
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = model_plot_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

        # ROC Curve Plot (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                n_classes = len(np.unique(self.y_test))
                plt.figure(figsize=(7, 6))
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] ROC curve plot saved to: {roc_path}")
                else:
                    # Multiclass ROC curve
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(self.y_test == i, y_pred_proba[:, i])
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Multiclass ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] Multiclass ROC curve plot saved to: {roc_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot ROC curve: {str(e)}")

        # Create a central 'plots' directory and subdirectory for this model
        plots_root = self.output_dir / "plots"
        plots_root.mkdir(exist_ok=True)
        model_plot_dir = plots_root / f"{model_name.replace(' ', '_').lower()}"
        model_plot_dir.mkdir(exist_ok=True)

        # Confusion Matrix Plot
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = model_plot_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

        # ROC Curve Plot (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                n_classes = len(np.unique(self.y_test))
                plt.figure(figsize=(7, 6))
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] ROC curve plot saved to: {roc_path}")
                else:
                    # Multiclass ROC curve
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(self.y_test == i, y_pred_proba[:, i])
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Multiclass ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    roc_path = model_plot_dir / 'roc_curve.png'
                    plt.savefig(roc_path, dpi=200)
                    plt.close()
                    print(f"[INFO] Multiclass ROC curve plot saved to: {roc_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot ROC curve: {str(e)}")
    
    
    def compare_models(self):
        """
        Compare all trained models and identify the best performing one.
        
        Creates a comparison table and visualizations.
        """
        print("\n" + "="*70)
        print("[FINAL] Model Comparison")
        print("="*70)
        
        if not self.model_results:
            print("[WARNING] No models have been trained yet.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': results['test_accuracy'],
                'Test F1-Score': results['test_f1_score'],
                'Test ROC-AUC': results['test_roc_auc'] if results['test_roc_auc'] is not None else np.nan,
                'CV Accuracy (mean±std)': f"{results['cv_accuracy_mean']:.4f}±{results['cv_accuracy_std']:.4f}",
                'CV F1-Score (mean±std)': f"{results['cv_f1_mean']:.4f}±{results['cv_f1_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test F1-Score', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model based on F1-Score
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n[BEST MODEL] {best_model_name}")
        print(f"  Test Accuracy: {comparison_df.iloc[0]['Test Accuracy']:.4f}")
        print(f"  Test F1-Score: {comparison_df.iloc[0]['Test F1-Score']:.4f}")
        if not pd.isna(comparison_df.iloc[0]['Test ROC-AUC']):
            print(f"  Test ROC-AUC:  {comparison_df.iloc[0]['Test ROC-AUC']:.4f}")
        
        # Save comparison to CSV
        output_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Model comparison saved to: {output_path}")
        
        # Create visualization
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    
    def _plot_model_comparison(self, comparison_df):
        """
        Create visualization comparing model performances.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame containing model comparison metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Test Accuracy
        axes[0].barh(comparison_df['Model'], comparison_df['Test Accuracy'], color='skyblue')
        axes[0].set_xlabel('Accuracy', fontsize=12)
        axes[0].set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlim([0, 1])
        
        # Plot 2: Test F1-Score
        axes[1].barh(comparison_df['Model'], comparison_df['Test F1-Score'], color='lightcoral')
        axes[1].set_xlabel('F1-Score', fontsize=12)
        axes[1].set_title('Test Set F1-Score', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        
        # Plot 3: Test ROC-AUC
        if not comparison_df['Test ROC-AUC'].isna().all():
            axes[2].barh(comparison_df['Model'], comparison_df['Test ROC-AUC'], color='lightgreen')
            axes[2].set_xlabel('ROC-AUC', fontsize=12)
            axes[2].set_title('Test Set ROC-AUC', fontsize=14, fontweight='bold')
            axes[2].set_xlim([0, 1])
        else:
            axes[2].text(0.5, 0.5, 'ROC-AUC Not Available', 
                        ha='center', va='center', fontsize=14)
            axes[2].set_title('Test Set ROC-AUC', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Model comparison plot saved to: {output_path}")
        plt.close()
    
    
    def save_results(self):
        """
        Save all model results and trained models to disk.
        """
        print("\n" + "="*70)
        print("[SAVING] Saving Results and Models")
        print("="*70)
        
        # Save model results as JSON
        results_path = self.output_dir / 'model_results.json'
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, results in self.model_results.items():
            json_results[model_name] = {
                'test_accuracy': float(results['test_accuracy']),
                'test_f1_score': float(results['test_f1_score']),
                'test_roc_auc': float(results['test_roc_auc']) if results['test_roc_auc'] is not None else None,
                'cv_accuracy_mean': float(results['cv_accuracy_mean']),
                'cv_accuracy_std': float(results['cv_accuracy_std']),
                'cv_f1_mean': float(results['cv_f1_mean']),
                'cv_f1_std': float(results['cv_f1_std'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"[SUCCESS] Model results saved to: {results_path}")
        
        # Save trained models using joblib
        import joblib
        
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        

        # Save all models
        for model_name, model in self.trained_models.items():
            model_path = models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_path)
            print(f"[SUCCESS] {model_name} saved to: {model_path}")

        # Save the best model separately as best_model.pkl
        # Find best model by highest test F1-score
        best_model_name = max(self.model_results, key=lambda k: self.model_results[k]['test_f1_score'])
        best_model = self.trained_models[best_model_name]
        best_model_path = models_dir / "best_model.pkl"
        joblib.dump(best_model, best_model_path)
        print(f"[SUCCESS] Best model ({best_model_name}) saved to: {best_model_path}")
        
        # Save preprocessing objects
        preprocessing_path = models_dir / 'preprocessing.pkl'
        preprocessing_objects = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessing_objects, preprocessing_path)
        print(f"[SUCCESS] Preprocessing objects saved to: {preprocessing_path}")
    
    
    def run_full_pipeline(self, tune_hyperparameters=True):
        """
        Run the complete model training pipeline.
        
        Parameters:
        -----------
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning for all models (default: True)
        """
        # Start logging to file
        self.tee_output = TeeOutput(self.log_file_path)
        sys.stdout = self.tee_output
        
        print("\n" + "="*70)
        print("ALZHEIMER'S DISEASE PREDICTION - MODEL TRAINING PIPELINE")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {self.log_file_path}")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Handle missing values
            self.handle_missing_values()
            
            # Step 3: Detect and handle outliers
            self.detect_and_handle_outliers()
            
            # Step 4: Encode categorical features
            self.encode_categorical_features()
            
            # Step 5: Prepare features and target
            self.prepare_features_and_target()
            
            # Step 6: Split data
            self.split_data(test_size=0.2, stratify=True)
            
            # Step 7: Scale features
            self.scale_features(method='standard')
            
            # Train all models
            print("\n" + "="*70)
            print("[TRAINING] Starting Model Training")
            print("="*70)
            
            self.train_logistic_regression(tune_hyperparameters=tune_hyperparameters)
            self.train_svm(tune_hyperparameters=tune_hyperparameters)
            self.train_knn(tune_hyperparameters=tune_hyperparameters)
            self.train_naive_bayes()
            self.train_decision_tree(tune_hyperparameters=tune_hyperparameters)
            self.train_random_forest(tune_hyperparameters=tune_hyperparameters)
            self.train_xgboost(tune_hyperparameters=tune_hyperparameters)
            self.train_lightgbm(tune_hyperparameters=tune_hyperparameters)
            self.train_catboost(tune_hyperparameters=tune_hyperparameters)
            self.train_stacking_ensemble()
            
            # Compare all models
            self.compare_models()
            
            # Save results
            self.save_results()
            
            print("\n" + "="*70)
            print("[COMPLETED] Pipeline Execution Completed Successfully!")
            print("="*70)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n[INFO] Complete log saved to: {self.log_file_path}")
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {str(e)}")
            raise
        
        finally:
            # Restore stdout and close log file
            if self.tee_output:
                sys.stdout = self.tee_output.terminal
                self.tee_output.close()
                print(f"\n[INFO] Training log saved to: {self.log_file_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block for the model training pipeline.
    
    Usage:
    ------
    1. The data path is configured in config.py (PROCESSED_DATA_PATH)
    2. Set tune_hyperparameters=True for full hyperparameter tuning (slower but better results)
       or tune_hyperparameters=False for quick training with default parameters
    3. Run the script: python model_trainer.py
    """
    
    # Configuration - using paths from config.py
    TARGET_COLUMN = 'Diagnosis'
    TUNE_HYPERPARAMETERS = True  # Set to False for faster execution
    RANDOM_STATE = 42
    
    # Initialize trainer
    trainer = AlzheimerModelTrainer(
        data_path=PROCESSED_DATA_PATH,
        target_column=TARGET_COLUMN,
        random_state=RANDOM_STATE
    )
    
    # Run the complete pipeline
    trainer.run_full_pipeline(tune_hyperparameters=TUNE_HYPERPARAMETERS)
    
    print("\n[INFO] All results and models have been saved to 'src/model training/output/'")
    print("[INFO] Training log with all terminal output has been saved as a .txt file")
    print("[INFO] You can now use the trained models for predictions on new data.")
