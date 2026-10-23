# ADNI Predictor

Predicting Alzheimer's disease from **real ADNI participants**. This folder replaces the synthetic-data pipeline in `src/` (which is kept unchanged for the existing API and frontend).

| Task | Question | Label |
|---|---|---|
| `CN_AD` | Can baseline data separate cognitively normal from Alzheimer's dementia? | baseline diagnosis CN (0) vs AD (1) |
| `sMCI_pMCI` | Which people with MCI will develop dementia? | pMCI (1): dementia within 36 months; sMCI (0): MCI at every visit for at least 36 months |

## ⚠️ Data rules (ADNI Data Use Agreement)

- **Never commit or push ADNI data or ADNI results.** `data/`, `output/` and every `*.csv` in this folder are git-ignored. Check with `git status` before every commit.
- Every team member who uses the data needs their own approved ADNI account.
- Results stay local until the manuscript has been submitted to the ADNI Data and Publications Committee.

## Folder layout

```
adni_predictor/
├── data/            ← put the IDA downloads here (git-ignored)
├── output/          ← everything the scripts produce (git-ignored)
│   ├── eda/                 Table 1, missingness, exploratory figures
│   ├── models/CN_AD/        results, comparison table, ROC, importance figures
│   ├── models/sMCI_pMCI/
│   └── summary.md
├── src/
│   ├── adni_config.py       paths and constants
│   ├── cohort.py            baseline cohorts, labels, FreeSurfer scan matching
│   ├── features.py          feature sets
│   ├── evaluate.py          models, repeated CV, external test, metrics
│   ├── plots.py             figure style (PNG 300 dpi + PDF)
│   ├── eda.py               exploratory analysis
│   └── run_models.py        model comparison
└── tests/                   synthetic-data unit tests (no ADNI data)
```

## Data needed in `data/`

| File (from IDA → Study Data) | Used for |
|---|---|
| `ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]_*.csv` (under _Archive) | diagnosis, demographics, APOE4, cognitive tests, summary volumes |
| `UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1,GO,2,3,4]-UCSFFSX7_*.csv` | regional brain volumes and cortical thickness |
| `...Data Dictionary...-DATADIC_*.csv` (optional) | readable FreeSurfer region names in figures |

Another location works too: set `ADNI_DATA_DIR`. A folder `adni_data/` at the repo root is picked up automatically if `data/` does not exist.

## Run

From the repo root:

```bash
pip install -r adni_predictor/requirements.txt
python adni_predictor/src/eda.py                     # exploratory analysis -> output/eda/
python adni_predictor/src/run_models.py              # model comparison -> output/models/, output/summary.md
python adni_predictor/src/run_models.py --repeats 2  # quick run
python -m unittest discover -s adni_predictor/tests -v
```

## Method

- **One row per participant** (baseline visit), so every split is subject-level: nobody is in training and test at once.
- **Within ADNI1:** stratified 5-fold cross-validation repeated 10 times (1.5T, one protocol). Imputation and scaling are fit on the training folds only.
- **External test:** train on all of ADNI1, test once on participants enrolled in ADNIGO/2/3 (3T, other years and sites).
- **Model selection uses CV only.** Per feature set the model with the best mean CV ROC-AUC is marked `selected`. Picking a model by its external score would leak the test set.
- **Models:** logistic regression, SVM (RBF), k-NN, MLP neural network, random forest, extra trees, XGBoost, LightGBM. Hyperparameters are fixed in advance; logistic regression and SVM tune `C` with an inner CV on the training fold.
- **Feature sets:** demographics + APOE4 (`demo`), cognitive tests (`cog`), ADNIMERGE FreeSurfer summary volumes (`mri_merge`, within ADNI1 only because their FreeSurfer version changes by phase), FreeSurfer 7 regional volumes and thickness (`mri_fs7`). Volumes are divided by intracranial volume.
- Only participants with a FreeSurfer 7 baseline scan (within 180 days) are used, so every feature set is evaluated on the same people.

## Known limitations

- **Cognitive tests are circular for `CN_AD`.** ADNI diagnoses are defined with MMSE, CDR and memory tests, so those results (AUC ≈ 1.0) are a ceiling, not a finding. For `sMCI_pMCI` they are partly circular too (conversion is judged with the same tests).
- **Age confound:** ADNI3 controls are about 5 years younger than ADNI3 AD participants (see `output/eda/fig_age_CN_AD`).
- **Population shift:** ADNI1 has only late MCI, ADNIGO/2/3 add early MCI, so conversion rates differ between training and external test.
- Most FreeSurfer 7 rows are not yet manually quality-controlled by UCSF (`STATUS = partial`); only rows rated `Fail` are dropped.
