"""Central paths and constants for the ADNI pipeline.

ADNI data is covered by the ADNI Data Use Agreement: it must NEVER be committed or pushed.
The data folder is referenced by path only and is git-ignored.

Override with environment variables:
    ADNI_DATA_DIR    folder with the IDA downloads (default: PredictorV2/data, git-ignored;
                     falls back to <repo>/adni_data if that exists and data/ does not)
    ADNI_OUTPUT_DIR  where results are written (default: PredictorV2/output, git-ignored)

Output layout (one folder per stage, one subfolder per task):
    output/1_data_analysis/<task>/{tables,figures}
    output/2_model_comparison/<task>/<fixed|tuned>_hyperparameters/{tables,figures,hyperparameters}
    output/3_explainability/<task>/<mode>/<feature set>/{shap,lime,permutation_importance}
    output/summary.md
"""

from __future__ import annotations

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]         # PredictorV2/src
PROJECT_DIR = SRC_DIR.parent                          # PredictorV2
REPO_ROOT = PROJECT_DIR.parent                        # alzheimerpredictor


def _default_data_dir() -> Path:
    data = PROJECT_DIR / "data"
    legacy = REPO_ROOT / "adni_data"
    return legacy if legacy.is_dir() and not data.is_dir() else data


DATA_DIR = Path(os.environ.get("ADNI_DATA_DIR", _default_data_dir())).resolve()
OUTPUT_DIR = Path(os.environ.get("ADNI_OUTPUT_DIR", PROJECT_DIR / "output")).resolve()

STAGES = {"analysis": "1_data_analysis", "models": "2_model_comparison", "explain": "3_explainability"}
TASKS = ["CN_AD", "sMCI_pMCI"]

# IDA file names carry a download date, so files are found by pattern.
ADNIMERGE_GLOB = "ADNIMERGE - Key ADNI tables merged into one table [[]*.csv"
FS7_GLOB = "*UCSFFSX7*.csv"
DATADIC_GLOB = "*DATADIC*.csv"

# Labels
CONVERSION_MONTHS = 36          # pMCI: Dementia within 36 months; sMCI: MCI at every visit for >= 36 months
MAX_SCAN_GAP_DAYS = 180         # FreeSurfer scan must be this close to the baseline exam date

# Evaluation
TRAIN_PHASE = "ADNI1"
TEST_PHASES = ["ADNIGO", "ADNI2", "ADNI3"]
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
TUNE_ITER = 20                  # random-search candidates per inner search
TUNE_INNER_SPLITS = 3
N_JOBS = int(os.environ.get("ADNI_N_JOBS", -1))   # parallel CV folds (-1 = all cores)


def out_dir(stage: str, *parts: str) -> Path:
    """output/<numbered stage>/<parts...>, created on first use."""
    path = OUTPUT_DIR.joinpath(STAGES[stage], *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_file(pattern: str, data_dir: Path | None = None, exclude: str | None = "Dictionary") -> Path:
    data_dir = Path(data_dir or DATA_DIR)
    hits = sorted(p for p in data_dir.rglob(pattern) if not exclude or exclude not in p.name)
    if not hits:
        raise FileNotFoundError(
            f"No file matching '{pattern}' under {data_dir}. Set ADNI_DATA_DIR to the folder with the IDA downloads."
        )
    if len(hits) > 1:
        raise FileExistsError(f"Several files match '{pattern}' under {data_dir}: {[h.name for h in hits]}")
    return hits[0]
