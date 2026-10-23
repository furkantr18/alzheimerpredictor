# ADNI Predictor

Predicting Alzheimer's disease from **real ADNI participants**. This folder is the new, ADNI-based version of the project. The old synthetic-data pipeline in `src/` is kept unchanged for the existing API and frontend.

| Task | Question | Label |
|---|---|---|
| `CN_AD` | Can baseline data separate cognitively normal people from Alzheimer's dementia? | baseline diagnosis CN (0) vs AD (1) |
| `sMCI_pMCI` | Which people with MCI will develop dementia? | pMCI (1): dementia within 36 months; sMCI (0): MCI at every visit for at least 36 months |

## ⚠️ Data rules (ADNI Data Use Agreement)

- **Never commit or push ADNI data or ADNI results.** `data/`, `output/` and every `*.csv` in this folder are git-ignored. Check `git status` before every commit.
- Everyone who uses the data needs their own approved ADNI account.
- Results stay local until the manuscript has been submitted to the ADNI Data and Publications Committee.

## Folder layout

```
PredictorV2/
├── run.py                       ← the only script you run
├── data/                        ← put the IDA downloads here (git-ignored)
├── src/
│   ├── common/                  config (paths, constants), cohort + labels, features, figure style
│   ├── data_analysis/           eda.py (Table 1, figures), statistics.py (tests, effect sizes, ANOVA)
│   ├── training/                models.py (8 models + search spaces), evaluate.py (metrics, CV), run_models.py
│   └── explainability/          explain.py (SHAP, LIME, permutation importance)
├── tests/                       unit tests on synthetic data (no ADNI data)
└── output/                      everything the scripts produce (git-ignored)
    ├── 1_data_analysis/
    │   ├── CN_AD/          tables/ (Table 1, group comparison, missingness)  figures/
    │   ├── sMCI_pMCI/      tables/  figures/
    │   └── all_participants/   ANOVA across CN / EMCI / LMCI / AD
    ├── 2_model_comparison/
    │   └── <task>/
    │       ├── fixed_hyperparameters/                   summary.md, tables/, figures/, hyperparameters/
    │       ├── fixed_hyperparameters_anova_selection/   (after `models --select`)
    │       └── tuned_hyperparameters/                   (after `models --tune`)
    └── 3_explainability/
        └── <task>/<mode>/<feature set>/   shap/  lime/  permutation_importance/
```

## Data needed in `data/`

| File (IDA → Study Data) | Used for |
|---|---|
| `ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]_*.csv` (under _Archive) | diagnosis, demographics, APOE4, cognitive tests, summary volumes |
| `UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1,GO,2,3,4]-UCSFFSX7_*.csv` | regional brain volumes and cortical thickness |
| `...Data Dictionary...-DATADIC_*.csv` (optional) | readable FreeSurfer region names in figures |

Another location works too: set `ADNI_DATA_DIR`. A folder `adni_data/` at the repo root is picked up automatically if `data/` does not exist.

## Run

From the repo root:

```bash
pip install -r PredictorV2/requirements.txt

python PredictorV2/run.py analysis          # 1. Table 1, statistics, exploratory figures   (~1 min)
python PredictorV2/run.py models            # 2. 8 models x 7 feature sets, 5-fold CV x 10   (~15-30 min)
python PredictorV2/run.py explain           # 3. SHAP, LIME, permutation importance        (~5 min)
python PredictorV2/run.py all               # 1 + 2 + 3

python PredictorV2/run.py models --repeats 2                      # quick run
python PredictorV2/run.py models --tune                           # nested-CV tuning (slow, see below)
python PredictorV2/run.py models --select                         # ANOVA-FDR feature selection, no leakage
python PredictorV2/run.py explain --select                        # explain the --select models
python PredictorV2/run.py models --tasks CN_AD --models xgb logreg --feature-sets demo+mri_eng

python -m unittest discover -s PredictorV2/tests -v
```

## What each stage produces

**1. Data analysis** (`output/1_data_analysis/`)
- `table1_characteristics.csv`: participants per phase and label (age, sex, education, APOE4, MMSE).
- `group_comparison.csv`: per feature, mean (SD) per group, Welch t-test, Mann-Whitney U, Hedges' g, chi-square for sex and APOE4, Benjamini-Hochberg FDR q-values. **Descriptive only**: these p-values are never used to choose model features, because selecting features on all participants before CV leaks test information.
- `anova_baseline_diagnosis.csv`: one-way ANOVA and Kruskal-Wallis across CN / EMCI / LMCI / AD with eta squared and FDR.
- Figures: participant counts, age by phase (age confound), markers by label, engineered markers, effect sizes, correlations, markers across the four diagnoses.

**2. Model comparison** (`output/2_model_comparison/<task>/<mode>/`)
- `summary.md`: the main results table.
- `tables/results_all_metrics.csv`: per model and feature set, CV mean and SD plus external values for ROC-AUC, PR-AUC, balanced accuracy, accuracy, precision, recall, specificity, F1, macro-F1, FPR, FNR, Brier score and log loss.
- `tables/selected_models.csv`: the CV-selected model per feature set with all metrics and hyperparameters.
- `tables/comparison_roc_auc.csv`, `tables/results_per_repeat.csv`.
- `hyperparameters/final_params_trained_on_ADNI1.csv`, and with `--tune` also `best_params_per_outer_fold.csv`.
- Figures: model comparison, ROC-AUC heatmap, ROC curves, precision-recall curves, calibration, confusion matrices.

**3. Explainability** (`output/3_explainability/<task>/<mode>/<feature set>/`): the CV-selected model is refit on ADNI1 and explained on the external participants. SHAP beeswarm and bar plots, LIME for three participants (a correct positive, a correct negative and the most confident error), and permutation importance.

## Method

- **One row per participant** (baseline visit), so every split is subject-level.
- CV folds run in parallel on all CPU cores (set `ADNI_N_JOBS=2` to use fewer).
- **Within ADNI1:** stratified 5-fold CV repeated 10 times (1.5T, one protocol). Imputation and scaling are fit on the training folds only.
- **External test:** train on all of ADNI1, test once on participants enrolled in ADNIGO/2/3 (3T, other years and sites).
- **Model selection uses CV only.** Per feature set the model with the best mean CV ROC-AUC is `selected`. Choosing by the external score would leak the test set.
- **Models:** logistic regression, SVM (RBF), k-NN, MLP neural network, random forest, extra trees, XGBoost, LightGBM.
- **Hyperparameters:** by default fixed in advance. With `--tune`, nested CV: a random search (20 candidates, 3-fold inner CV) runs inside every outer training fold, so the outer test fold never influences tuning. The external model is tuned on all of ADNI1. Tuning runs 3 repeats on the 3 main feature sets by default, because it is roughly 60 times slower.
- **Feature selection without leakage (`--select`):** an ANOVA F-test (sklearn `f_classif`) with Benjamini-Hochberg FDR at 0.05 is a step *inside* every model pipeline. It is refit on the training part of every fold, so the test fold never influences which features are kept. `tables/feature_selection_frequency.csv` reports how often each feature was kept across the 50 folds (a stability measure) and whether it is kept in the final ADNI1 model. The p-values in `1_data_analysis` are computed on all participants and are therefore descriptive only: never pick model features from them.
- **Feature sets:** `demo` (age, sex, education, APOE4), `cog` (cognitive tests), `mri_merge` (ADNIMERGE summary volumes, ADNI1 only), `mri_fs7` (about 200 FreeSurfer 7 regional volumes and thicknesses), `mri_eng` (10 literature-based markers, below). Volumes are divided by intracranial volume (ICV).
- Only participants with a FreeSurfer 7 baseline scan (within 180 days) are used, so every feature set is evaluated on the same people.

### Engineered MRI markers (`mri_eng`)

| Feature | Definition |
|---|---|
| hippocampus_icv | total hippocampal volume / ICV |
| hippocampus_asymmetry | \|L − R\| / mean(L, R) |
| hippocampal_occupancy | hippocampus / (hippocampus + inferior lateral ventricle), Heister et al. 2011 |
| amygdala_icv | total amygdala volume / ICV |
| mtl_volume_icv | hippocampus + amygdala + entorhinal + parahippocampal volume / ICV |
| entorhinal_thickness | mean entorhinal thickness (mm) |
| parahippocampal_thickness | mean parahippocampal thickness (mm) |
| ad_signature_thickness | area-weighted thickness of entorhinal, fusiform, inferior and middle temporal cortex, Jack et al. 2015 |
| ventricles_icv | lateral + inferior lateral ventricles / ICV |
| total_gm_icv | total grey matter / ICV |

Note: `Ventricles_icv` and `Hippocampus_icv` (capitalised) are the ADNIMERGE versions; the lower-case ones are computed from FreeSurfer 7.

## Known limitations

- **Cognitive tests are circular for `CN_AD`.** ADNI diagnoses are defined with MMSE, CDR and memory tests, so those results (AUC ≈ 1.0) are a ceiling, not a finding. For `sMCI_pMCI` they are partly circular too (conversion is judged with the same tests).
- **Age confound:** ADNI3 controls are about 5 years younger than ADNI3 AD participants (`fig_age`).
- **Population shift:** ADNI1 has only late MCI, while ADNIGO/2/3 add early MCI, so conversion rates differ between training and external test.
- Most FreeSurfer 7 rows are not yet manually quality-controlled by UCSF (`STATUS = partial`); only rows rated `Fail` are dropped.
