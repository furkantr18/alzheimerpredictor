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

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEST_MODEL_PATH, PROCESSED_DATA_PATH


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
            
            # Load normalization metadata
            metadata_path = Path(__file__).parent / "data" / "processed" / "processing_metadata.json"
            with open(metadata_path, 'r') as f:
                self.normalization_metadata = json.load(f)
            print(f"[SUCCESS] Normalization metadata loaded from: {metadata_path}")
            
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
        
        print("[INFO] Step 1: Normalizing test data...")
        # First, normalize the raw data using the same parameters as training
        data = self.normalize_test_data(data)
        
        print("[INFO] Step 2: Dropping non-predictive columns...")
        # Drop non-predictive ID columns if present
        id_columns = ['PatientID', 'DoctorInCharge', 'Diagnosis']
        for col in id_columns:
            if col in data.columns:
                print(f"[INFO]   - Dropping column: {col}")
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
        
        print("[INFO] Step 3: Selecting and ordering features to match training...")
        # Select and order features to match training
        data = data[self.feature_names]
        
        print("[INFO] Step 4: Applying scaler transformation...")
        # Scale features using the saved scaler
        scaled_data = self.scaler.transform(data)
        
        print("[INFO] Preprocessing complete!")
        return scaled_data
    
    
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
    
    
    def predict_batch(self, patient_data_path, output_path=None):
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
        
        # Make predictions for each patient
        results = []
        for idx, row in df.iterrows():
            try:
                result = self.predict(row.to_dict(), return_probabilities=True)
                result['PatientID'] = patient_ids[idx]
                results.append(result)
            except Exception as e:
                print(f"[WARNING] Failed to predict for patient {patient_ids[idx]}: {e}")
                results.append({
                    'PatientID': patient_ids[idx],
                    'diagnosis': 'ERROR',
                    'error': str(e)
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['PatientID', 'diagnosis', 'prediction_code', 'risk_score']
        if 'probabilities' in results_df.columns:
            # Expand probabilities into separate columns
            prob_df = pd.DataFrame(results_df['probabilities'].tolist())
            prob_df.columns = [f'prob_{col}' for col in prob_df.columns]
            results_df = pd.concat([results_df[cols], prob_df], axis=1)
        
        # Save or print results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"[SUCCESS] Predictions saved to: {output_path}")
        else:
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
            predictor.predict_batch(args.batch, output_path=args.output)
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
