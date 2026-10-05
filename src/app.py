from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import pandas as pd
import os
import subprocess
import uuid
import sys
from pathlib import Path
import threading
from typing import List, Dict, Any
import math
import numpy as np
import json

app = FastAPI(
    title="Alzheimer Predictor API",
    description="Lightweight API for health checks and prediction endpoints.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,  # Starlette >= 1.x rejects PNA preflights unless this is set
)

# Chrome Private Network Access: an https page (oracleapex.com) calling localhost
# needs this header on the preflight and on the actual response.
@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

logger = logging.getLogger("alzheimer_api")

# Absolute paths so the API works from any working directory
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# ---------------------------
# Mapping dictionary for CSV column names
# ---------------------------
COLUMN_MAP = {
    'PatientID': 'PATIENT_ID',
    'Age': 'AGE',
    'Gender': 'GENDER',
    'Ethnicity': 'ETHNICITY',
    'EducationLevel': 'EDUCATION_LEVEL',
    'BMI': 'BMI',
    'Smoking': 'SMOKING',
    'AlcoholConsumption': 'ALCOHOL_CONSUMPTION',
    'PhysicalActivity': 'PHYSICAL_ACTIVITY',
    'DietQuality': 'DIET_QUALITY',
    'SleepQuality': 'SLEEP_QUALITY',
    'FamilyHistoryAlzheimers': 'FAMILY_HISTORY_ALZHEIMERS',
    'CardiovascularDisease': 'CARDIOVASCULAR_DISEASE',
    'Diabetes': 'DIABETES',
    'Depression': 'DEPRESSION',
    'HeadInjury': 'HEAD_INJURY',
    'Hypertension': 'HYPERTENSION',
    'SystolicBP': 'SYSTOLIC_BP',
    'DiastolicBP': 'DIASTOLIC_BP',
    'CholesterolTotal': 'CHOLESTEROL_TOTAL',
    'CholesterolLDL': 'CHOLESTEROL_LDL',
    'CholesterolHDL': 'CHOLESTEROL_HDL',
    'CholesterolTriglycerides': 'CHOLESTEROL_TRIGLYCERIDS',
    'MMSE': 'MMSE',
    'FunctionalAssessment': 'FUNCTIONAL_ASSESSMENT',
    'MemoryComplaints': 'MEMORY_COMPLAINTS',
    'BehavioralProblems': 'BEHAVIORAL_PROBLEMS',
    'ADL': 'ADL',
    'Confusion': 'CONFUSION',
    'Disorientation': 'DISORIENTATION',
    'PersonalityChanges': 'PERSONALITY_CHANGES',
    'DifficultyCompletingTasks': 'DIFFICULTY_COMPLETING_TASKS',
    'Forgetfulness': 'FORGETFULLNESS',
    'Diagnosis': 'DIAGNOSIS'
}

# Other spellings sent by APEX (EXPORT_PATIENTS_PKG, page 2 form JS, GET_PATIENT_DATA)
COLUMN_ALIASES = {
    'FAMILY_HISTORY_ALZHEMERS': 'FamilyHistoryAlzheimers',    # EXPORT_PATIENTS_PKG
    'CHOLESTEROL_TRYGLYCERIDS': 'CholesterolTriglycerides',   # page 2 form JS (item P2_CHOLESTEROL_TRYGLYCERIDS)
    'FORGETFULNESS': 'Forgetfulness',                         # package, form JS, GET_PATIENT_DATA
}

# Columns that are not model features
NON_FEATURE_COLUMNS = {'PatientID', 'Diagnosis'}

# In-memory queue for incoming patient entries (not persisted to CSV)
# This is intentionally process-local and kept lightweight. Use a lock
# to avoid races when FastAPI runs multiple threads in the same process.
INCOMING_LIST: List[Dict[str, Any]] = []
INCOMING_LIST_LOCK = threading.Lock()

# ---------------------------
# RECEIVE DATA FROM APEX
# ---------------------------
@app.post("/patients_data")
async def patients_data(req: Request):
    """
    Receives JSON from APEX (single dict or list of dicts) and saves to CSV.
    """
    try:
        data = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Request body is not valid JSON."})
    print("RAW JSON:", data)

    # Ensure data is a list of dicts
    if isinstance(data, dict) and data:
        data = [data]

    # Reject empty bodies and rows without any value
    if not isinstance(data, list) or not data:
        return JSONResponse(status_code=400, content={"error": "Empty request body: send one patient object or a list of patient objects."})
    empty_rows = [i for i, row in enumerate(data)
                  if not isinstance(row, dict) or all(v is None or str(v).strip() == "" for v in row.values())]
    if empty_rows:
        return JSONResponse(status_code=400, content={"error": f"Rows without any values (0-based index): {empty_rows[:20]}"})

    # Convert to DataFrame and rename columns according to mapping (plus aliases)
    df = pd.DataFrame(data)
    rename_map = {v: k for k, v in COLUMN_MAP.items()}
    rename_map.update(COLUMN_ALIASES)
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # Warn when model features are still missing after the rename (predict.py would zero-fill them)
    missing_features = [k for k in COLUMN_MAP if k not in NON_FEATURE_COLUMNS and k not in df.columns]
    if missing_features:
        logger.warning("Missing model features after rename (will be zero-filled): %s", missing_features)
        print(f"WARNING: missing model features after rename (will be zero-filled): {missing_features}")

    # Ensure folder exists
    folder_path = SRC_DIR / "data" / "incomingData"
    os.makedirs(folder_path, exist_ok=True)

    # CSV file path
    csv_file = os.path.join(folder_path, "patients_data.csv")

    # Save to CSV -- overwrite existing file so the new upload replaces previous data
    df.to_csv(csv_file, index=False)
    print(f"Saved incoming patients file: {csv_file} ({len(df)} rows) - overwritten")

    return {"received": len(data), "missing_features": missing_features, "message": f"Data saved to {csv_file} successfully!"}


@app.post("/patient_data")
async def patient_data(req: Request):
    """
    req.json() ile gelen JSON'u dict veya list olarak alıyoruz.
    Bu sayede APEX'ten gelen tek obje veya array JSON sorunsuz işlenir.
    """
    data = await req.json()
    print("RAW JSON:", data)

    # Ensure data is a list of dicts
    if isinstance(data, dict):
        data = [data]

    # Assign a random PATIENT_ID if not provided
    for entry in data:
        if not entry.get('PATIENT_ID') and not entry.get('PatientID'):
            new_id = uuid.uuid4().hex[:12]
            entry['PATIENT_ID'] = new_id
            entry['PatientID'] = new_id

    # Append entries to the in-memory list (do NOT persist to CSV)
    with INCOMING_LIST_LOCK:
        for entry in data:
            # Normalize incoming external column names to internal if needed
            # (keep original keys as-is so predictor can accept them)
            INCOMING_LIST.append(entry.copy())
        current_len = len(INCOMING_LIST)
    assigned = [d.get('PATIENT_ID') or d.get('PatientID') for d in data]
    print(f"Appended {len(data)} incoming patient(s) to in-memory queue. Assigned IDs: {assigned}. Queue length: {current_len}")
    return {"received": len(data), "assigned_ids": assigned, "queue_length": current_len, "message": "Data appended to in-memory queue (not saved to CSV)."}

# ---------------------------
# SEND DATA TO APEX AFTER PREDICTION
# ---------------------------
@app.get("/patients_data/send")
def send_patients_data():
    """
    Runs predict.py to generate test_eval_results.csv,
    then returns that CSV as JSON to APEX.
    """
    try:
        prediction_script = str(SRC_DIR / "prediction" / "predict.py")
        output_csv = os.path.join(os.path.dirname(prediction_script), "test_eval_results.csv")

        # Run the prediction script with a timeout and capture output for debugging
        # (sys.executable = the venv python that runs this API)
        try:
            proc = subprocess.run([sys.executable, prediction_script], check=True, capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as e:
            return {"error": f"Prediction script failed (return code {e.returncode})", "stderr": e.stderr, "stdout_tail": (e.stdout or "")[-2000:]}
        except subprocess.TimeoutExpired as e:
            return {"error": "Prediction script timed out", "details": str(e)}

        # Check if CSV exists
        if not os.path.exists(output_csv):
            return {"error": "test_eval_results.csv not found after running prediction.", "stdout": getattr(proc, 'stdout', None), "stderr": getattr(proc, 'stderr', None)}

        # Read CSV and convert to JSON
        try:
            df = pd.read_csv(output_csv)
            data = df.to_dict(orient='records')  # list of dicts
        except Exception as e:
            return {"error": f"Failed to read CSV: {str(e)}"}

        # Sanitize values (replace NaN/inf with None) for JSON compliance
        def _sanitize_val(v):
            try:
                if v is None:
                    return None
                if isinstance(v, float):
                    if math.isnan(v) or not math.isfinite(v):
                        return None
                    # convert numpy floats
                if isinstance(v, (np.floating, np.integer)):
                    if np.isnan(v) or not np.isfinite(v):
                        return None
                    return v.item()
                return v
            except Exception:
                return None

        def _sanitize_record(rec: dict) -> dict:
            out = {}
            for k, val in rec.items():
                if isinstance(val, dict):
                    out[k] = _sanitize_record(val)
                elif isinstance(val, list):
                    out[k] = [_sanitize_val(x) if not isinstance(x, dict) else _sanitize_record(x) for x in val]
                else:
                    out[k] = _sanitize_val(val)
            return out

        data = [_sanitize_record(r) for r in data]
        print(f"Sent batch predictions to APEX: {len(data)} rows")
        return {"data": data}
    except Exception as e:
        # Catch-all to avoid uncaught exceptions causing 500 responses
        import traceback
        tb = traceback.format_exc()
        return {"error": "Unexpected server error in send_patients_data", "details": str(e), "trace": tb}
    
@app.get("/patient_data/send")
def send_patient_data():
    """
    If exactly one patient is queued in-memory, run single-instance prediction
    in-process (using the `AlzheimerPredictor` in `src/prediction/predict.py`)
    and return the prediction to APEX. Otherwise fall back to batch behavior.
    """
    prediction_script = str(SRC_DIR / "prediction" / "predict.py")

    # Check in-memory queue first
    try:
        with INCOMING_LIST_LOCK:
            queued = list(INCOMING_LIST)
        if len(queued) == 1:
            # Run single-instance prediction in-process
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("prediction_predict", os.path.abspath(prediction_script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                Predictor = getattr(mod, 'AlzheimerPredictor')
                predictor = Predictor()
                row = queued[0]
                result = predictor.predict(row, return_probabilities=True)
                out = row.copy()
                out.update(result)
                # Print outgoing payload to console for debugging / APEX visibility
                try:
                    print("Outgoing to APEX (single):", json.dumps(out, default=str, ensure_ascii=False))
                except Exception:
                    print("Outgoing to APEX (single):", out)
                # Remove the predicted item from the queue
                with INCOMING_LIST_LOCK:
                    try:
                        INCOMING_LIST.pop(0)
                    except Exception:
                        pass
                # Return the single prediction wrapped as list for APEX
                print(f"Sent single prediction to APEX for patient: {out.get('PATIENT_ID') or out.get('PatientID')}")
                return {"data": [out]}
            except Exception as e:
                return {"error": f"Single-instance prediction failed: {str(e)}"}

        # Otherwise, fall back to existing batch behavior
        res = send_patients_data()
        # Print the outgoing batch payload (truncated if very large)
        try:
            print("Outgoing to APEX (batch):", json.dumps(res, default=str, ensure_ascii=False))
        except Exception:
            print("Outgoing to APEX (batch):", res)
        try:
            if isinstance(res, dict) and 'data' in res and isinstance(res['data'], list):
                print(f"Sent batch predictions to APEX via fallback: {len(res['data'])} rows")
        except Exception:
            pass
        return res

    except Exception as e:
        return {"error": f"Unexpected error in send_patient_data: {str(e)}"}
# ---------------------------
# SIMPLE GET HANDLER
# ---------------------------
@app.get("/patients_data_handler")
def handler():
    return {"status": "patients_data endpoint is working"}
