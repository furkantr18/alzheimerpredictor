"""
Alzheimer's Disease Prediction - Inference Script
==================================================
This script loads a trained model and makes predictions on new patient data.

Usage:
------
1. For single patient prediction:
   python src/predict.py --patient_data "path/to/patient_data.csv"

2. For interactive prediction:
   python src/predict.py --interactive

3. For batch prediction:
   python src/predict.py --batch "path/to/patients_batch.csv"
"""
# hasnt been tested yet - please test before using 
# no test dataset ....

import pandas as pd
import numpy as np
import joblib
import argparse
import json
from pathlib import Path
import sys
import os
import sqlite3

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEST_MODEL_PATH, PROCESSED_DATA_PATH, PREDICTIONS_DIR, DEFAULT_BATCH_PREDICTIONS_CSV


class AlzheimerPredictor:
    """
    A class for making Alzheimer's disease predictions on new patient data.
    
    This class handles:
    - Loading trained models and preprocessing objects
    - Preprocessing new patient data
    - Making predictions with probability scores
    - Interpreting and formatting results
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor by loading the trained model and preprocessing objects.
        
        Parameters:
        -----------
        model_path : str or Path, optional
            Path to the trained model file. If None, uses BEST_MODEL_PATH from config.
        """
        self.model_path = model_path or BEST_MODEL_PATH
        self.model_dir = Path(self.model_path).parent
        
        print("[INFO] Loading trained model and preprocessing objects...")
        
        try:
            # Load the trained model
            self.model = joblib.load(self.model_path)
            print(f"[SUCCESS] Model loaded from: {self.model_path}")
            
            # Load the scaler
            scaler_path = self.model_dir / "scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print(f"[SUCCESS] Scaler loaded from: {scaler_path}")
            
            # Load preprocessing objects (label encoders, feature names, etc.)
            preprocessing_path = self.model_dir / "preprocessing.pkl"
            preprocessing = joblib.load(preprocessing_path)
            self.label_encoders = preprocessing['label_encoders']
            self.feature_names = preprocessing['feature_names']
            print(f"[SUCCESS] Preprocessing objects loaded from: {preprocessing_path}")
            
            # Get diagnosis classes
            if 'target' in self.label_encoders:
                self.diagnosis_classes = self.label_encoders['target'].classes_
            else:
                self.diagnosis_classes = ['Cognitive Normal', 'Alzheimer\'s Disease']
            
            print(f"[INFO] Model ready for prediction. Classes: {self.diagnosis_classes}")
            
        except FileNotFoundError as e:
            print(f"[ERROR] Required file not found: {e}")
            print("[ERROR] Please ensure the model has been trained first by running model_trainer.py")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    
<<<<<<< HEAD
=======
    def normalize_test_data(self, data):
        """
        Normalize test data using the same normalization parameters from training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw test data to normalize
            
        Returns:
        --------
        pd.DataFrame
            Normalized test data
        """
        data = data.copy()
        
        # Normalize numeric columns using min-max scaling
        for col, params in self.normalization_metadata['numeric'].items():
            if col in data.columns and params['scaled']:
                min_val = params['min']
                max_val = params['max']
                if min_val != max_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)
        
        # Binary columns (0/1) - ensure they are integers
        for col in self.normalization_metadata['binary_cols']:
            if col in data.columns:
                data[col] = data[col].astype(int)
        
        return data
    
    def _post_normalization_pipeline(self, data: pd.DataFrame) -> np.ndarray:
        """
        Apply steps after normalization: drop IDs, encode categoricals (if any),
        ensure feature order, and apply the saved scaler.

        Parameters:
        -----------
        data : pd.DataFrame
            Normalized data.

        Returns:
        --------
        np.ndarray
            Scaled feature matrix ready for the model.
        """
        df = data.copy()

        # Drop non-predictive ID columns if present
        id_columns = ['PatientID', 'DoctorInCharge', 'Diagnosis']
        for col in id_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Encode categorical features using saved label encoders (if any)
        for col, encoder in self.label_encoders.items():
            if col != 'target' and col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Unknown category -> fallback to first known class
                    df[col] = encoder.transform([encoder.classes_[0]])[0]

        # Ensure all expected features exist
        missing_features = set(self.feature_names) - set(df.columns)
        for feature in missing_features:
            df[feature] = 0

        # Order columns
        df = df[self.feature_names]

        # Apply scaler
        X = self.scaler.transform(df)
        return X

    
>>>>>>> b006ba8 (Update predict)
    def preprocess_input(self, patient_data):
        """
        Preprocess patient data to match the format expected by the model.
        
        Parameters:
        -----------
        patient_data : pd.DataFrame or dict
            Patient data to preprocess
            
        Returns:
        --------
        np.ndarray
            Preprocessed and scaled feature array ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Make a copy to avoid modifying original data
        data = patient_data.copy()
        
<<<<<<< HEAD
        # Drop non-predictive ID columns if present
        id_columns = ['PatientID', 'DoctorInCharge', 'Diagnosis']
        for col in id_columns:
            if col in data.columns:
                data = data.drop(columns=[col])
        
        # Encode categorical features using saved label encoders
        for col, encoder in self.label_encoders.items():
            if col != 'target' and col in data.columns:
                try:
                    data[col] = encoder.transform(data[col].astype(str))
                except ValueError as e:
                    print(f"[WARNING] Unknown category in {col}: {e}")
                    # Handle unknown categories by using the most frequent class
                    data[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all expected features are present
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            print(f"[WARNING] Missing features: {missing_features}")
            print("[INFO] Setting missing features to 0")
            for feature in missing_features:
                data[feature] = 0
        
        # Select and order features to match training
        data = data[self.feature_names]
        
        # Scale features using the saved scaler
        scaled_data = self.scaler.transform(data)
        
        return scaled_data
=======
        print("[INFO] Step 1: Normalizing test data...")
        data_norm = self.normalize_test_data(data)
        print("[INFO] Step 2-4: Dropping IDs, ordering features, scaling...")
        X = self._post_normalization_pipeline(data_norm)
        print("[INFO] Preprocessing complete!")
        return X
>>>>>>> b006ba8 (Update predict)
    
    
    def predict(self, patient_data, return_probabilities=True):
        """
        Make prediction for patient data.
        
        Parameters:
        -----------
        patient_data : pd.DataFrame, dict, or str
            Patient data (DataFrame, dict, or path to CSV file)
        return_probabilities : bool
            Whether to return probability scores (default: True)
            
        Returns:
        --------
        dict
            Dictionary containing prediction results
        """
        # Load data from file if path is provided
        if isinstance(patient_data, str):
            patient_data = pd.read_csv(patient_data)
            print(f"[INFO] Loaded patient data from: {patient_data}")
        
        # Preprocess the input
        X = self.preprocess_input(patient_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        diagnosis = self.diagnosis_classes[prediction]
        
        # Get probability scores if model supports it
        probabilities = None
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.diagnosis_classes, proba)
            }
        
        # Prepare result
        result = {
            'diagnosis': diagnosis,
            'prediction_code': int(prediction),
            'probabilities': probabilities,
            'risk_score': float(probabilities[diagnosis]) if probabilities else None
        }
        
        return result
    
    
    def predict_batch(self, patient_data_path, output_path=None, threshold: float = 0.5, save_normalized: str | None = None, save_sqlite: str | None = None, min_confidence: float = 0.1):
        """
        Make predictions for multiple patients from a CSV file.
        
        Parameters:
        -----------
        patient_data_path : str
            Path to CSV file containing patient data
        output_path : str, optional
            Path to save predictions. If None, prints to console
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        print(f"[INFO] Loading batch data from: {patient_data_path}")
        df = pd.read_csv(patient_data_path)
        
        print(f"[INFO] Making predictions for {len(df)} patients...")
        
        # Store patient IDs if present
        patient_ids = df['PatientID'].tolist() if 'PatientID' in df.columns else list(range(len(df)))
        
        # Preprocess all patients at once (more efficient)
        try:
            # Normalize once so we can optionally persist normalized data
            df_norm = self.normalize_test_data(df)

            # Optionally save normalized data to CSV
            if save_normalized:
                out_csv = Path(save_normalized)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                df_norm.to_csv(out_csv, index=False)
                print(f"[SUCCESS] Normalized test data saved to CSV: {out_csv}")

            # Optionally save normalized data to SQLite
            if save_sqlite:
                db_path = Path(save_sqlite)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(db_path) as conn:
                    df_norm.to_sql('normalized_test', conn, if_exists='replace', index=False)
                print(f"[SUCCESS] Normalized test data saved to SQLite table 'normalized_test' in: {db_path}")

            # Continue with post-normalization pipeline
            X = self._post_normalization_pipeline(df_norm)
            
            # Make predictions for all patients
            if hasattr(self.model, 'predict_proba'):
                proba_all = self.model.predict_proba(X)
                # Determine index of Alzheimer's class
                if isinstance(self.diagnosis_classes, (list, np.ndarray)) and len(self.diagnosis_classes) == 2:
                    try:
                        ad_index = list(self.diagnosis_classes).index("Alzheimer's Disease")
                    except ValueError:
                        # Fallback assume positive class is 1
                        ad_index = 1
                else:
                    ad_index = 1
                # Apply configurable threshold on Alzheimer's probability
                predictions = (proba_all[:, ad_index] >= float(threshold)).astype(int)
            else:
                predictions = self.model.predict(X)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = proba_all
            
            # Build results
            results = []
            for idx, pred in enumerate(predictions):
                diagnosis = self.diagnosis_classes[pred]
                result = {
                    'PatientID': patient_ids[idx],
                    'diagnosis': diagnosis,
                    'prediction_code': int(pred),
                }
                
                if probabilities is not None:
                    proba = probabilities[idx]
                    # Pull Alzheimer's probability consistently
                    try:
                        ad_index
                    except NameError:
                        # Determine Alzheimer's index if not set (should be set earlier)
                        try:
                            ad_index = list(self.diagnosis_classes).index("Alzheimer's Disease")
                        except Exception:
                            ad_index = 1
                    ad_proba = float(proba[ad_index])
                    result['risk_score'] = float(proba[pred])
                    result['prob_ad'] = ad_proba
                    # Confidence margin from 0.5 boundary
                    result['margin'] = abs(ad_proba - 0.5)
                    result['uncertain'] = result['margin'] < float(min_confidence)
                    result['probabilities'] = {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.diagnosis_classes, proba)
                    }
                else:
                    result['risk_score'] = None
                    result['probabilities'] = None
                    result['prob_ad'] = None
                    result['margin'] = None
                    result['uncertain'] = None
                
                results.append(result)
                
        except Exception as e:
            print(f"[ERROR] Batch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # If ground truth labels exist, compute evaluation metrics and include them
        if 'Diagnosis' in df.columns:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
                y_true = df['Diagnosis'].astype(int).to_numpy()
                y_pred = results_df['prediction_code'].astype(int).to_numpy()

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)

                # Attach ground truth for saved CSV
                results_df.insert(1, 'y_true', y_true)

                print("\n" + "="*70)
                print("EVALUATION (from predict.py - using provided Diagnosis column)")
                print("="*70)
                print(f"Accuracy : {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall   : {rec:.4f}")
                print(f"F1-score : {f1:.4f}")
                print("Confusion Matrix [ [TN FP] [FN TP] ]:")
                print(cm)

                print("\nCLASSIFICATION REPORT:")
                print(classification_report(y_true, y_pred, target_names=["Cognitive Normal", "Alzheimer's Disease"]))

                # If probabilities available, suggest threshold by quick sweep for F1
                if 'prob_ad' in results_df.columns and results_df['prob_ad'].notnull().all():
                    sweep = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
                    best_t, best_f1 = None, -1
                    print("\nTHRESHOLD SWEEP (F1 on provided batch):")
                    for t in sweep:
                        y_pred_s = (results_df['prob_ad'].to_numpy() >= t).astype(int)
                        f1s = f1_score(y_true, y_pred_s, zero_division=0)
                        print(f"  t={t:.2f} -> F1={f1s:.3f}")
                        if f1s > best_f1:
                            best_f1, best_t = f1s, t
                    if best_t is not None:
                        print(f"Suggested threshold on this batch: {best_t:.2f} (F1={best_f1:.3f}). Use --threshold {best_t:.2f} to apply.")
            except Exception as e:
                print(f"[WARNING] Could not compute evaluation metrics: {e}")
        
        # Reorder columns
        cols = ['PatientID', 'diagnosis', 'prediction_code', 'risk_score']
        if 'probabilities' in results_df.columns:
            # Expand probabilities into separate columns
            prob_df = pd.DataFrame(results_df['probabilities'].tolist())
            prob_df.columns = [f'prob_{col}' for col in prob_df.columns]
            # Keep additional diagnostics if present
            extra_cols = []
            for c in ['prob_ad', 'margin', 'uncertain']:
                if c in results_df.columns:
                    extra_cols.append(c)
            results_df = pd.concat([results_df[cols + extra_cols], prob_df], axis=1)
        
        # Determine default output path if not provided
        save_path = output_path
        if not save_path:
            # Ensure directory exists
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            save_path = DEFAULT_BATCH_PREDICTIONS_CSV

        # Save results always, and also print a compact view
        try:
            results_df.to_csv(save_path, index=False)
            print(f"[SUCCESS] Predictions saved to: {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save predictions to {save_path}: {e}")

        # Print a readable summary to console
        print("\n" + "="*70)
        print("BATCH PREDICTION RESULTS")
        print("="*70)
        print(results_df.to_string(index=False))
        
        return results_df
    
    
    def interactive_prediction(self):
        """
        Interactive mode for entering patient data and getting predictions.
        """
        print("\n" + "="*70)
        print("INTERACTIVE ALZHEIMER'S DISEASE PREDICTION")
        print("="*70)
        print("\nPlease enter patient information:")
        print("(Enter values for the following features)\n")
        
        # Get sample data to show feature names and expected ranges
        try:
            sample_data = pd.read_csv(PROCESSED_DATA_PATH)
            # Drop non-predictive columns
            for col in ['PatientID', 'DoctorInCharge', 'Diagnosis']:
                if col in sample_data.columns:
                    sample_data = sample_data.drop(columns=[col])
        except:
            sample_data = None
        
        patient_data = {}
        
        for feature in self.feature_names:
            if sample_data is not None and feature in sample_data.columns:
                # Show feature statistics
                min_val = sample_data[feature].min()
                max_val = sample_data[feature].max()
                mean_val = sample_data[feature].mean()
                prompt = f"{feature} (range: {min_val:.2f} - {max_val:.2f}, avg: {mean_val:.2f}): "
            else:
                prompt = f"{feature}: "
            
            while True:
                try:
                    value = input(prompt).strip()
                    if value:
                        patient_data[feature] = float(value)
                        break
                    else:
                        print("[ERROR] Please enter a value")
                except ValueError:
                    print("[ERROR] Please enter a valid number")
        
        # Make prediction
        print("\n[INFO] Making prediction...")
        result = self.predict(patient_data, return_probabilities=True)
        
        # Display results
        self._display_result(result)
    
    
    def _display_result(self, result):
        """
        Display prediction result in a formatted way.
        
        Parameters:
        -----------
        result : dict
            Prediction result dictionary
        """
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        
        diagnosis = result['diagnosis']
        risk_score = result.get('risk_score')
        
        print(f"\n🔍 Diagnosis: {diagnosis}")
        
        if risk_score is not None:
            print(f"📊 Confidence Score: {risk_score*100:.2f}%")
        
        if result.get('probabilities'):
            print("\n📈 Probability Breakdown:")
            for class_name, prob in result['probabilities'].items():
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {class_name:30s} {bar} {prob*100:5.2f}%")
        
        # Risk interpretation
        if risk_score is not None:
            print("\n💡 Interpretation:")
            if diagnosis == 'Alzheimer\'s Disease' or 'Alzheimer' in diagnosis:
                if risk_score >= 0.8:
                    print("  ⚠️  HIGH RISK: Strong indicators of Alzheimer's disease detected.")
                    print("     Recommendation: Consult with a neurologist immediately.")
                elif risk_score >= 0.6:
                    print("  ⚠️  MODERATE RISK: Some indicators of Alzheimer's disease present.")
                    print("     Recommendation: Schedule a comprehensive neurological evaluation.")
                else:
                    print("  ⚠️  LOW-MODERATE RISK: Mild indicators detected.")
                    print("     Recommendation: Monitor symptoms and follow up with healthcare provider.")
            else:
                if risk_score >= 0.8:
                    print("  ✅ LOW RISK: Strong indicators of cognitive health.")
                    print("     Recommendation: Continue regular health monitoring.")
                elif risk_score >= 0.6:
                    print("  ✅ MODERATE-LOW RISK: Generally healthy cognitive indicators.")
                    print("     Recommendation: Maintain healthy lifestyle and regular checkups.")
                else:
                    print("  ⚠️  BORDERLINE: Indicators are not strongly conclusive.")
                    print("     Recommendation: Consider additional testing for clarity.")
        
        print("\n" + "="*70)
        print("⚕️  DISCLAIMER: This is a predictive model and NOT a medical diagnosis.")
        print("   Always consult with qualified healthcare professionals.")
        print("="*70 + "\n")


def main():
    """
    Main function to handle command-line arguments and run predictions.
    """
    parser = argparse.ArgumentParser(
        description='Alzheimer\'s Disease Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python src/predict.py --interactive
  
  # Single patient from CSV
  python src/predict.py --patient data/new_patient.csv
  
  # Batch prediction
  python src/predict.py --batch data/patients_batch.csv --output results.csv
  
  # Use specific model
  python src/predict.py --interactive --model "src/model training/output/trained_models/random_forest.pkl"
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode (enter patient data manually)')
    parser.add_argument('--patient', '-p', type=str,
                       help='Path to CSV file with single patient data')
    parser.add_argument('--batch', '-b', type=str,
                       help='Path to CSV file with multiple patients')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path for batch predictions (CSV format)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Probability threshold for Alzheimer class when model supports predict_proba (default: 0.5)')
    parser.add_argument('--save-normalized', type=str,
                       help='Optional path to save normalized test data (CSV)')
    parser.add_argument('--save-sqlite', type=str,
                       help="Optional path to a SQLite DB file to save normalized test data in table 'normalized_test'")
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Mark predictions as uncertain when |p(AD)-0.5| < min-confidence (default: 0.1)')
    parser.add_argument('--model', '-m', type=str,
                       help='Path to specific model file (default: best_model.pkl)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = AlzheimerPredictor(model_path=args.model)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize predictor: {e}")
        return 1
    
    # Run appropriate prediction mode
    try:
        if args.interactive:
            predictor.interactive_prediction()
        elif args.patient:
            result = predictor.predict(args.patient, return_probabilities=True)
            predictor._display_result(result)
        elif args.batch:
            predictor.predict_batch(
                args.batch,
                output_path=args.output,
                threshold=args.threshold,
                save_normalized=args.save_normalized,
                save_sqlite=args.save_sqlite,
                min_confidence=args.min_confidence,
            )
        else:
            print("[ERROR] Please specify a prediction mode:")
            print("  --interactive  : Enter patient data manually")
            print("  --patient FILE : Predict for single patient from CSV")
            print("  --batch FILE   : Predict for multiple patients from CSV")
            parser.print_help()
            return 1
        
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
