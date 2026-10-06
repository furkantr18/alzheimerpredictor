# MRI image integration: code, honest evaluation, API

> **Erratum (2026-10-05, branch `feature/mri-improvement`).** The patient-ID reconstruction described below had a bug: the slice-25 row and the `a (b)` tail files were attached to the wrong patient (1,441 of 6,400 slices, always within the same class). So the "patient-level" split used here was not fully patient-level: some slices of a test patient belonged to a real person whose other slices were in training. Labels were never wrong. The "3.2% near-twin slices" finding was caused by this bug (after the fix: 1 pair). The numbers in this report are therefore slightly optimistic and are superseded by the nested-CV results in `docs/MRI_IMPROVEMENT_REPORT.md`. Details: `docs/EXPERIMENT_PROTOCOL.md`, amendment 4.

Backend repo (furkantr18/alzheimerpredictor), branch **`feature/mri-integration`** (local only, not pushed).
Everything here was run on this machine (Windows 11, Python 3.13.7, NVIDIA RTX 3050 Laptop 4 GB) on 2026-10-05.
The marks used: **[verified]** = run here on the real data; **[synthetic]** = run only on synthetic images; **[unverified]** = reasoning or estimate, not run.

## 0. Results at a glance [verified at the time; PRE-FIX, SUPERSEDED: see the erratum above and `docs/MRI_IMPROVEMENT_REPORT.md`]

Test set of the honest split = 960 ORIGINAL slices of 30 patients never seen in training (ModerateDemented: 1 patient, 32 slices). Evaluated once (`final_evaluation.py`). 4 classes; chance balanced accuracy 0.25; always answering "NonDemented" gives accuracy 0.50.

Figure: run `python src/imgProcessing/make_figures.py` to regenerate `fig_leakage_vs_honest.png` in the git-ignored output folder (not committed).

| Setting | Model | Test accuracy | Balanced acc. | Macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| **Naive split** (Yasemin's setting: `archive`, random per image, copies + same patients) | ResNet-18, same recipe | **0.999** | 0.999 | 0.999 | 1.000 |
| Naive split | PCA+LogReg (fixed) | 0.649 | 0.657 | 0.649 | 0.863 |
| Image-level split, originals only (same patients across splits) | PCA+LogReg (fixed) | 0.586 | 0.684 | 0.618 | – |
| **Patient-level split (honest)** | **ResNet-18, 3 seeds** | **0.554 ± 0.015** | **0.509 ± 0.009** | **0.484 ± 0.015** | **0.817 ± 0.013** |
| Patient-level split | EfficientNet-B0, 3 seeds | 0.564 ± 0.006 | 0.506 ± 0.019 | 0.490 ± 0.026 | 0.811 ± 0.004 |
| Patient-level split | ResNet-50, 3 seeds | 0.576 ± 0.026 | 0.478 ± 0.027 | 0.475 ± 0.031 | 0.797 ± 0.019 |
| Patient-level split | best classical (SVM-RBF on PCA) | 0.555 | 0.433 | 0.427 | 0.736 |
| Patient-level split | PCA+LogReg (fixed) | 0.460 | 0.376 | 0.366 | 0.695 |
| Patient-level split | majority class | 0.500 | 0.250 | 0.167 | – |

**Bottom line (pre-fix, superseded; corrected nested-CV estimates: 4-class 0.38, 3-class 0.52, binary 0.72 patient-level macro-F1):** the same network that scores 99.9% under the naive split reaches about **51% balanced accuracy / 0.48 macro-F1** on unseen patients. That is clearly above chance (0.25) but far from usable. Served model `final_resnet18_s42` (chosen on validation before the test run): test accuracy 0.568, balanced accuracy 0.520, macro-F1 0.499, ROC-AUC 0.816; subject-cluster bootstrap 95% CI for macro-F1 **0.41–0.61** (only 30 test patients, so every number carries roughly ±0.1).

Per-class (served model, test): MildDemented recall 0.81 / precision 0.32; ModerateDemented recall 0.25 (8 of 32 slices of the single test patient); NonDemented F1 0.74; VeryMildDemented recall 0.34. Most errors are confusions among the neighbouring stages Mild ↔ VeryMild ↔ Non. When the slice probabilities of each test patient are averaged, the served model gets 18 of 30 patients right (0.60). Across the 9 runs this patient-level accuracy ranges 0.57–0.80, from only 30 patients.

### Why Yasemin's results could look "extremely low" or "extremely high"
- Her own DL code, run unchanged on a 4,000-image random subset of `archive` (naive split, 6 epochs), gets test macro-F1 **0.81** and is still rising [verified, `_backend/mri_baseline/yasemin_original_dl_subset.log`]. Her original classical trainer on the same subset gets 0.55–0.68 [verified]. So her code does not produce "extremely low" numbers on this data. On the full naive data it would go towards 99% because of leakage, as our naive run shows.
- Plausible causes of her low numbers (**unverified**, ask her): a different folder layout (e.g. `archive (1)/data` read as 2 classes `train`/`val`, or `combined_images` as 1 class); an early or partial run; or results from the classical pipeline. The honest answer to "how good is MRI classification here" is the patient-level row above (about 0.48 macro-F1), and it is low for real reasons: 2D slices, 4 fine-grained stages, few patients per class (Moderate = 2).

## 1. Source of the code (Phase 0)

| Step | What happened |
|---|---|
| Safety | `git fetch --all --prune`; new branch `feature/mri-integration` from `main` (03e5999). `main` is untouched. |
| First commit `fb5c888` | The two backend fixes from the earlier local rebuild (`src/app.py`, `model_trainer.py`, see `BACKEND_CHANGES.md`) were uncommitted on `main`'s working tree; they are committed here so the branch is the working state. |
| Search | Only `origin/yasamin_temp` (commit `ae3bce0`, "img processing", 2026-03-06) touches image code. All other branches and PR refs (`furkan_temp`, `gamze_temp`, `pr/1..10`): 0 image commits, 0 files with cv2/PIL/torch. |
| Taken, unchanged (commit `2b39691`) | `src/imgProcessing/{preprocess_mri, mri_model_trainer, mri_dl_trainer, predict_mri, predict_mri_dl}.py`, `src/data/README.md`, the MRI section of `README.md`, her `.gitignore` lines (`src/data/img_processed`, `src/data/img_unprocessed`, `src/imgProcessing/output/`). |
| **Not** taken | Deletion of `src/data/processed/{processing_metadata.json, processed_data.csv, normalize_raw.py}` (breaks tabular predictions: 8/25 diagnoses change, verified earlier). Changes to `src/data/incomingData/patients_data.csv` and `src/prediction/test_eval_results.*` (unrelated run outputs). Her `requirements.txt` edits (torch/opencv would become mandatory for the tabular app) → moved to a separate `requirements-image.txt`. |
| Rewritten afterwards | Every script was then fixed or rewritten in later commits (diffs in git). Her preprocessing pipeline `preprocess_mri.py` is kept as an optional offline tool (rotation bug fixed). |

Commits on the branch (oldest first): `fb5c888`, `2b39691`, `b0e0425`, `e858665`, `ff782d4`, `e2283c5`. Nothing pushed. (Copy of the local working report.)

## 2. Datasets (Phase 1)

### Which folder is which (confidence: high, ~95%)

| Folder | Dataset | Evidence |
|---|---|---|
| `datasets/archive (1)` | **uraninjo/augmented-alzheimer-mri-dataset-v2** | `data/train/<4 classes>` 33,984 RGB JPGs, UUID names, 200x190 or 180x180; `data/val/<4 classes>` 6,400 grayscale 176x208 with the original names (`mildDem12.jpg`, `26 (19).jpg`). val counts 896/64/3,200/2,240 = the well-known original "Alzheimer's Dataset (4 class of images)". |
| `datasets/archive` | **aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented** | `combined_images/<4 classes>` 44,000 JPGs = all 40,384 images of `archive (1)` (**pixel-identical**, same names) + 3,616 extra `aug_<k>_*` copies (3,472 ModerateDemented, 144 MildDemented) to balance the classes to 10,000/10,000/12,800/11,200. |

**Yasemin's code expects `archive`**: her README names uraninjo, but her class counts (44,000; 12,800/11,200/10,000/10,000) and the flat `<class>/*.jpg` layout her scripts read are exactly `archive/combined_images`. Pointed at `archive (1)/data` her code would have treated `train`/`val` as two classes.

### Inventory [verified] (`output/inventory/`, `inventory_summary.json`, sample grids `samples_*.png`)
- Classes in both: MildDemented, ModerateDemented, NonDemented, VeryMildDemented (same names → mapping is the identity).
- No unreadable files (84,384 read).
- Exact duplicates: `archive (1)/train` has 195 groups of pixel-identical images (658 images, no label conflicts). `archive (1)/val`: none.
- **The shipped train/val split of `archive (1)` is leaky by construction**: `train` consists of augmented copies of the `val` images (visual check of the grids; 7,855 train images even have a pHash identical to a val image). So `archive (1)`'s own split cannot be used for evaluation.
- Artefacts: brightness augmentation turned the black background grey in part of the copies (handled in preprocessing); copies are zoomed/stretched (200x190) versus originals (176x208).
- Mean intensity differs by class (originals: NonDemented 74.3, VeryMild 70.1, Moderate 69.6, Mild 67.4). This is consistent with atrophy (more dark CSF) but is also a global cue a model could use; see the global-feature baseline in §4.

### Patient (subject) IDs: reconstructed [verified at the time, but WRONG for the slice-25 row and the tail files: see the erratum]
The datasets ship no patient IDs, but the original file names encode them (`subject_ids.py`):
- `<class>Dem<i>`: slices are stored slice-major, so subject = i mod P and slice = i div P, with P = 28 (Mild), 2 (Moderate), 100 (Non), 70 (VeryMild). Image i and i+P correlate 0.99 versus 0.89 for i and i+1.
- `<a> (<b>)`: slice a (26..32) of subject b.
- Result: **200 subjects x 32 slices = 6,400**; 194 subjects have exactly 32 slices; per class, one subject has 33 and one 31 (an off-by-one in the original naming), so these 3 pairs are merged into one group each → 197 groups. Pixel check: 98.8% of same-subject neighbouring-slice pairs are more similar than the 99th percentile of different-subject pairs.
- ModerateDemented has only **2 subjects** (64 slices). This alone limits what any model can learn or prove about that class.

### External test set: not possible
Both folders are derived from the same 6,400 originals (all 6,400 are pixel-identical in both). There is no independent data, so no train-on-one / test-on-other experiment is meaningful. **Decision:** primary = `archive (1)` originals; `archive` is used only to reproduce Yasemin's naive setting.

## 3. Bugs from `_diagnosis_images/IMAGE_DIAGNOSIS.md`, re-verified with real OpenCV 5.0 / PyTorch 2.14 (Phase 2)

Log: `_backend/mri_baseline/reverify_bugs.log`. All re-checks ran Yasemin's ORIGINAL code from the snapshot.

| Bug | Status now | Fix |
|---|---|---|
| B1 deletion of `processing_metadata.json` breaks tabular predictions | Confirmed earlier (8/25 diagnoses change); **avoided**: files kept. 13/13 API tests pass and the 25 + 1 sample responses are byte-identical before/after [verified]. | not merged |
| B2 copies of one scan in train and test | **Confirmed on real data** (§4 leakage table, shuffled-label test). | subject-grouped split, originals only |
| B3 `--rotation-correction` crashes (numpy 2.x `eig` complex) | **Confirmed** with real cv2: `TypeError: ufunc 'arctan2' not supported`. | `eigh`; unit test |
| B4 upright brains rotated 90° | **Confirmed** with real cv2 (bbox 181x131 → 130x180). | align long axis to vertical, skip near-round masks; unit tests (upright untouched, 20° tilt straightened to < 2°) |
| B5 one unreadable file crashes training | **Confirmed**: `ValueError: ... [336, 337]`. | cache build and predictors skip + warn + count; unit tests |
| B6 memory/time on full data | Partly measured: original classical trainer on a 4,000-image subset took 303 s (PCA kept 722 components; SVC with `probability=True`). Peak RAM not captured (measurement failed) → full-data RAM still **[unverified]**. | randomized PCA(128) on 64x64 pixels; no `probability=True` |
| B7 train/inference preprocessing mismatch | By design in the old code. | ONE function `mri_preprocess.preprocess_gray()` used by cache, trainers, predictors and API; spec saved with each model |
| B8 relative paths | **Confirmed** (`FileNotFoundError` from another folder). | `mri_config.py` absolute paths + env/CLI overrides |
| B9 lambda transform + `num_workers>0` on Windows | **Confirmed** with torch 2.14: `AttributeError: Can't get local object ... <lambda>` (pickling). | no lambdas; batches built in the main process from a memmap cache |
| B10 torch on Python 3.13 | **Works**: torch 2.14.1+cu130, torchvision 0.29.1, numpy 2.5.3 unchanged, scikit-learn 1.7.2 pin kept. | `requirements-image.txt` |
| B11 small issues | Class from parent folder / stem collisions remain in the optional `preprocess_mri.py` (not used by the main pipeline); test set scored per model → now only `final_evaluation.py` touches test. | |
| B12 not integrated | Fixed: `/predict-mri`, `/predict-mri/explain`, `/mri/info`. | §6 |

Further problems found and fixed while working:
- **Class order / label mismatch risk**: old predictors used a separately fitted `LabelEncoder`; now one fixed `CLASS_NAMES` list is saved with every model; `predict_proba` maps model columns to that list even if a class was missing in training (unit test).
- **Seeds**: python/numpy/torch seeded; splits seeded and saved.
- **DL trainer**: ImageNet weights + ImageNet normalisation + gray→3 channels; augmentation only on train batches; `model.eval()` for val/test; class-weighted CE; AdamW with lower backbone LR; frozen backbone for the first epochs; warm-up + cosine LR; early stopping on validation loss; AMP on GPU; checkpoint every epoch + `--resume` (a resume bug — RNG state restored on GPU — was found by a real interruption and fixed).
- **Debug-before-tuning found two real problems**: (1) the overfit-a-tiny-batch check first failed (acc 0.28) because the warm-up schedule held the LR near 0 in that loop → fixed, now 32/32 images at 100% in 25 steps; (2) `channels_last` memory format made training 5-8x slower on this GPU/cuDNN (ResNet-50: 24 vs 190 img/s) → removed.
- **XGBoost** crashed in grouped CV folds that lack ModerateDemented (labels must be 0..k-1) → small label re-indexing wrapper.
- **Label smoothing + class weights** (PyTorch CE): the smoothing mass is weighted per class, and ModerateDemented's weight is ~35 (1 subject), so the model predicted ModerateDemented for 802 of 928 val images (val macro-F1 0.115). Not a code bug but a trap; label smoothing is not used.
- Preprocessing: grey backgrounds of brightness-augmented copies defeated the brain crop → background level (border median) is subtracted first (preprocessing v2; check `output/inventory/preprocessed_samples.png`).

## 4. The data, honestly (Phase 3)

### Leakage-aware split (`make_splits.py`, `output/splits/*.csv`, seed 42)
- **Honest split `split_subject.csv`**: 6,400 originals, units = 197 subject groups, per class 70/15/15. Train 4,512 images / 138 subjects, val 928 / 29, test 960 / 30. ModerateDemented: 1 subject in train, 1 in test, none in val. **0 subjects and 0 identical images shared** between train and val/test.
- **Dataset copies are not used** in the honest setting. I tried to trace each of the 33,521 unique copies to its source patient (ImageNet ResNet-50 embeddings, same class, mirrored originals, subject margin; `build_groups.py`). On the 46 copies whose source is written in the file name the matcher was right 93.5% of the time; pHash equality turned out not to be a reliable ground truth (different patients' slices can share a pHash). With ~6% of copies possibly assigned to the wrong patient, using them could re-introduce leakage, so training uses the 4,512 train originals with our own on-the-fly augmentation instead.
- The test split is read only by `final_evaluation.py`, which refuses a second run (`output/reports/test_used.json`).

### Leakage quantified on real data [verified] (`leakage_experiment.py`, test numbers from `final_results.json`)
The same fixed model (shared preprocessing → 64x64 pixels → StandardScaler → PCA(128) → LogisticRegression C=0.01, balanced), no tuning:

| Split | Real labels: test acc / macro-F1 | **Shuffled labels** (one random label per patient): test acc | Majority baseline acc |
|---|---|---|---|
| Naive (`archive`, per image) | 0.649 / 0.649 | **0.440** | 0.291 |
| Image-level, originals only | 0.586 / 0.618 | **0.407** | 0.500 |
| Patient-level (honest) | 0.460 / 0.366 | **0.255** | 0.500 |

With shuffled labels the images carry **no** disease information; the only way to score above chance (0.25) is to recognise copies or the same patient. The naive split gives 0.44 and the image-level split 0.41 → the leakage is real and large; the patient-level split gives 0.255 = chance, so the grouping removes it. (For the naive split, copies were given their patient by the ~93.5%-accurate matcher, so this test is conservative.) The deep model shows the same thing more strongly: 0.999 naive vs 0.55 patient-level.

**How sure am I that leakage is removed?** For identical images and dataset copies: certain (0 shared pixels; copies are not used). For patients: high (reconstructed IDs verified on pixels; shuffled-label test at chance). Remaining risk: the original dataset could contain the same person under two subject numbers (e.g. repeat scans). 61 of 1,888 val+test slices (3.2%) have a training slice as similar as neighbouring slices of one patient (r ≥ 0.98); these may be such repeats or simply alike slices at the top/bottom of the brain [verified count, cause unverified].

### Global-intensity cue [verified, validation only]
A classifier on 4 global numbers per slice (mean/std intensity inside the brain, brain area, dark-pixel share) reaches val balanced accuracy 0.34 (chance with 3 val classes 0.33) → global brightness is at most a weak cue; the dark-pixel share rises with stage (NonDemented 0.11 → Mild 0.18), consistent with more CSF/atrophy.

## 4b. Why so low, overfitting, hyper-parameters (Phase 4)

### Debug before tuning [verified]
- Overfit a tiny batch (32 train images, 8 per class): 100% accuracy in 25 steps after the fix described in §3. Labels and pipeline are wired correctly.
- Label alignment: per-class sample grids (`output/inventory/samples_*.png`) and preprocessed grids look consistent (class folder = label; no mixed folders).
- LR / optimiser / epochs / batch size / normalisation / frozen layers / class weights were checked through the search below; class imbalance is handled by class-weighted loss (ModerateDemented weight ≈ 35, see the label-smoothing trap in §3).

### Hyper-parameter search (ResNet-18, seed 42, validation = 928 originals of 29 unseen patients; selection by val macro-F1 at the best-val-loss epoch)
Baseline: 224 px, batch 32, lr 3e-4 (head) / 3e-5 (backbone), wd 1e-4, dropout 0.2, backbone frozen 1 epoch, augmentation strength 1, ≤25 epochs, early stop after 6 epochs without val-loss improvement.

| Variant | best epoch | val macro-F1 | val bal-acc | train acc (originals, at best epoch) |
|---|---|---|---|---|
| base | 2 | 0.481 | 0.483 | 0.662 |
| lr 1e-4 | 7 | 0.512 | 0.539 | 0.720 |
| lr 1e-3 | 2 | 0.456 | 0.459 | 0.685 |
| backbone LR = head LR | 2 | 0.368 | 0.423 | 0.587 |
| weight decay 1e-2 | 2 | 0.472 | 0.475 | 0.665 |
| dropout 0 / 0.5 | 2 / 5 | 0.474 / 0.497 | 0.484 / 0.496 | 0.662 / 0.789 |
| freeze 0 / 3 epochs | 2 / 5 | 0.502 / 0.499 | 0.511 / 0.517 | 0.693 / 0.718 |
| augmentation 0 / 2 | 2 / 5 | 0.464 / 0.502 | 0.464 / 0.511 | 0.798 / 0.731 |
| 160 px | 2 | 0.443 | 0.452 | 0.653 |
| batch 64 | 7 | 0.530 | 0.548 | 0.829 |
| label smoothing 0.1 | 3 | 0.115 | 0.085 | 0.130 (trap, §3) |
| **batch 64 + freeze 3 (chosen, greedy 2nd round)** | 7 | **0.532** | 0.554 | 0.745 |

Second round (greedy combination) did not beat 0.53 meaningfully (0.514–0.532). With 29 validation patients these differences (±0.03) are within noise; the three final seeds of the *same* chosen config spread 0.459–0.534 on validation, so the search mostly ruled out bad settings (high backbone LR, no augmentation, small images, label smoothing) rather than finding a clearly better one. Search tables: `output/reports/hp_search.csv`, `hp_search2.csv`.

### Final runs and overfitting [verified]
Chosen config for all three architectures, seeds 42/43/44 (ResNet-50 at batch 64 filled the 4 GB GPU and spilled into shared memory → slow, but results unaffected).

| Run | best epoch | train acc (orig.) | val acc | val macro-F1 | test acc | test bal-acc | test macro-F1 | test ROC-AUC | test patient-level acc |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 s42 | 7 | 0.741 | 0.584 | 0.534 | 0.568 | 0.520 | 0.499 | 0.816 | 0.600 |
| resnet18 s43 | 6 | 0.662 | 0.520 | 0.471 | 0.537 | 0.503 | 0.468 | 0.805 | 0.567 |
| resnet18 s44 | 6 | 0.710 | 0.528 | 0.459 | 0.557 | 0.505 | 0.485 | 0.831 | 0.600 |
| efficientnet_b0 s42 | 11 | 0.695 | 0.540 | 0.483 | 0.565 | 0.513 | 0.502 | 0.809 | 0.700 |
| efficientnet_b0 s43 | 13 | 0.717 | 0.525 | 0.471 | 0.570 | 0.522 | 0.507 | 0.816 | 0.667 |
| efficientnet_b0 s44 | 14 | 0.724 | 0.556 | 0.495 | 0.558 | 0.485 | 0.460 | 0.809 | 0.667 |
| resnet50 s42 | 7 | 0.658 | 0.525 | 0.437 | 0.549 | 0.450 | 0.454 | 0.778 | 0.633 |
| resnet50 s43 | 6 | 0.629 | 0.510 | 0.445 | 0.579 | 0.480 | 0.461 | 0.815 | 0.800 |
| resnet50 s44 | 8 | 0.682 | 0.539 | 0.452 | 0.600 | 0.504 | 0.510 | 0.797 | 0.733 |
| naive resnet18 s42 | 23 | 1.000 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | – |

- **Overfitting is the main pattern.** Every honest run reaches its lowest validation loss after 6–14 epochs. Training accuracy keeps rising (to 0.85–0.9) while validation loss rises and validation macro-F1 stays flat at about 0.5. Curves: `output/models/dl_runs/<run>/curves.png` (e.g. `final_resnet18_s42/curves.png`), history CSVs next to them.
- **Train/val/test gaps** at the chosen epoch: train ≈ 0.63–0.74 vs val ≈ 0.51–0.58 vs test ≈ 0.54–0.60 accuracy. Test is close to validation, so there is no sign that validation-based choices overfit the test.
- Classical models overfit harder: train macro-F1 0.74–1.00 vs val 0.37–0.45 (SVM 0.95 → 0.45; RF/KNN 1.00 → 0.37/0.41).
- Architecture differences (0.475–0.490 mean macro-F1) are smaller than the seed spread. Bigger models do not help with 138 training patients.

## 5b. What the model looks at (Phase 5) [verified, validation set, model `final_resnet18_s42`]

Grad-CAM on the last ResNet block (`mri_gradcam.py`; gallery `output/reports/gradcam/final_resnet18_s42/gallery.png`, per-image numbers `attention_per_image.csv`, summary `attention_stats.json`), all 928 validation slices:

| Region | Share of image area | Share of Grad-CAM mass | Ratio |
|---|---|---|---|
| Inner brain (brain mask minus a 7-px rim) | 0.475 | 0.718 | **1.51** |
| Brain edge ring (outer rim of the skull-stripped brain) | 0.103 | 0.090 | 0.87 |
| Background (black, outside the brain) | 0.422 | 0.192 | 0.45 |

- Attention concentrates on **central brain tissue: lateral ventricles and the periventricular white matter / deep grey matter**. No image has more than 50% of its attention outside the brain, and the brain border gets less than its area share. So there is **no sign of a border/skull/background shortcut**. There is no text or burnt-in label in these images.
- The cue looks like **ventricle size**: wrongly classified NonDemented slices with large ventricles are called VeryMild/Mild, and demented slices with small ventricles are called NonDemented. Ventricular enlargement is a real atrophy marker, but it also grows with normal ageing. This fits the low separation between neighbouring stages.
- The hippocampus (the key early-AD region) is not well represented in axial slices at these levels. On lower (temporal) slices, attention spreads sideways and partly to the image edge. Accuracy by slice position is flat (0.56–0.62 over 4 bands), so no slice band carries the result.
- Grad-CAM at 7x7 resolution is coarse; the background share partly comes from upsampling blur. These observations are qualitative and are not a validation of clinical reasoning.

## 5. Reproduce everything

Run from `_backend/alzheimerpredictor` with `.venv\Scripts\python.exe` (Windows paths with spaces are fine; datasets are read in place from `<project root>\datasets`, override with `MRI_DATASETS_DIR`; all outputs go to the git-ignored `src/imgProcessing/output/`, override with `MRI_OUTPUT_DIR`).

```
pip install -r requirements-image.txt --extra-index-url https://download.pytorch.org/whl/cu130   # once (~2 GB)
python src/imgProcessing/dataset_inventory.py      # 1 min: inventory, hashes, sample grids
python src/imgProcessing/inventory_report.py       # dataset comparison / overlap
python src/imgProcessing/subject_ids.py            # pseudo-patient ID check
python src/imgProcessing/build_cache.py            # 1 min: shared preprocessing of all 43,537 unique images
python src/imgProcessing/build_groups.py           # 3 min (GPU): copy->patient matching attempt + groups.csv
python src/imgProcessing/make_splits.py            # the three split CSVs
python src/imgProcessing/leakage_experiment.py     # naive vs image-level vs subject-level, shuffled labels
python src/imgProcessing/mri_model_trainer.py      # 6 classical models, grouped CV on train (~7 min)
python src/imgProcessing/run_experiments.py --stage all   # DL search + 9 final runs + 2 naive runs (~2.5 h on the RTX 3050)
python src/imgProcessing/data_checks.py            # cross-subject similarity, global-feature baseline (val)
python src/imgProcessing/final_evaluation.py       # THE single test evaluation
python src/imgProcessing/mri_gradcam.py --run <run>        # Grad-CAM gallery + attention statistics (val)
python src/imgProcessing/serve_model.py --run <run>        # copy the chosen model for the API
python -m unittest tests.test_mri_unit -v                  # 14 unit tests
python tests\test_mri_api.py --expect-model                # API tests (server running)
```
Long jobs log to `output/logs/` (`experiments.log`, `run_<name>.out`, per-run `train.log`) and resume after an interruption (rerun the same command).

## 6. Backend integration (Phase 6)

- `src/image_api.py`: separate FastAPI router. `src/app.py` gets one small block that mounts it inside `try/except`; all tabular code is unchanged. CORS and the Private-Network header middleware of the existing app apply automatically.
- `GET /mri/info` → `model_available`, class names, model name/version, test + validation metrics, dataset, disclaimer (200 even without a model).
- `POST /predict-mri` (multipart field `file`) → `predicted_class`, `confidence`, `probabilities` (4 classes), `model {name, version, kind}`, `input_quality_warning`, `disclaimer` = "Research prototype; not a medical diagnosis…".
- `POST /predict-mri/explain` → the same + `gradcam_png_base64` (deep model only; 409 for a classical model).
- Errors are JSON: 415 wrong content type, 400 empty / undecodable / smaller than 32 px, 413 larger than 10 MB (`MRI_MAX_UPLOAD_BYTES`), 503 no model or missing dependencies, 500 unexpected.
- Lazy loading: the model loads at the first MRI request; the tabular API starts as before and keeps working without torch/OpenCV or without a model. A model served later is picked up without restart.
- The served model lives in the git-ignored `src/imgProcessing/output/models/served/` with `served.json` (class names, preprocessing spec, library versions, metrics), `meta.json`, weights and the split CSV. Regenerate: train (§5) then `serve_model.py --run <run>`.
- `START_BACKEND.bat` is unchanged and works (same `.venv`).
- Late fusion with the tabular model is **not** implemented (future work): the two models are trained on different, unlinked people, so there is no patient with both inputs to learn or validate a fusion rule.

### Tests [verified]
- `tests/test_mri_unit.py`: 14/14 pass. They cover rotation (no crash, upright untouched, tilt fixed, round skipped), shared preprocessing (shape, determinism, grey background, RGB/bytes, undecodable → ValueError, 3-channel ImageNet tensor), unreadable files skipped in the cache build and the predictor, class order with a missing class, subject-ID parsing, split rules (Moderate 1/0/1, seeded), and GPU augmentation range.
- `tests/test_mri_api.py` against the server started with `START_BACKEND.bat`:
  - without a model: 9/9 (503 JSON, info says no model, all error codes);
  - with the served model: **15/15**: PNG and JPEG predictions, probabilities summing to 1, disclaimer, model name/version, CORS + Private-Network headers (same as the tabular endpoints), Grad-CAM PNG, and 400/413/415.
- Original tabular suite `_backend/tests/test_backend.py`: **13/13** before, after (no model), and after (with model). `sample_response.json` (25 patients) and `sample_response_form.json` are **byte-identical** to the pre-change versions (`_backend/mri_baseline/`).
- Example real call (an original NonDemented slice): `{"predicted_class":"NonDemented","confidence":0.8638,"probabilities":{...},"model":{"name":"resnet18 (final_resnet18_s42)","version":"20261005-1624","kind":"deep"},"disclaimer":"Research prototype; not a medical diagnosis. ..."}`. That slice belongs to a validation patient, so it is not a test of generalisation.
- The tabular API is reachable 1 s after `START_BACKEND.bat` starts; the MRI model loads lazily on the first MRI request.

## 7. Decisions log (where information was missing)

| # | Decision | Reason |
|---|---|---|
| D1 | `archive (1)` = uraninjo v2, `archive` = aryansinghal10; Yasemin used `archive` | folder structure, counts, file names, pixel identity (§2) |
| D2 | Primary data = the 6,400 originals of `archive (1)`; `archive` only for the naive reproduction | both sets contain the same originals; originals are the only images with traceable patients |
| D3 | No external test set | both folders derive from the same 6,400 images (all pixel-identical) |
| D4 | Patient IDs reconstructed from file names; 3 ambiguous subject pairs merged | dataset has no IDs; the naming pattern verified on pixels (98.8%); merging is the safe side |
| D5 | Dataset copies not used for training in the honest setting | copy→patient matching only ~93.5% accurate; our own augmentation on train originals instead |
| D6 | Split 70/15/15 by patient, seed 42; ModerateDemented 1 train / 0 val / 1 test patient | only 2 Moderate patients; test needs one, training needs one |
| D7 | Keep 4 classes (not merge Mild+Moderate) | matches the API/Yasemin's design; Moderate limitation reported instead |
| D8 | Shared preprocessing: background shift to black, brain bounding-box crop, letterbox to 224; no CLAHE/binning/edge blend | images are already skull-stripped/normalised by the dataset author; Yasemin's heavy pipeline stays optional (`preprocess_mri.py`) |
| D9 | Rotation correction fixed but not used in the main pipeline | slices are already upright; a wrong rotation would hurt |
| D10 | Classical features = 64x64 pixels → PCA(128) | 16k-pixel full SVD of the original needed many GB; 64 px keeps ~all PCA-useful structure |
| D11 | Hyper-parameters chosen on validation (seed 42), one factor at a time + greedy combination; selection metric val macro-F1 at best-val-loss epoch; early stopping on val loss | small, documented search as requested; the test set never used for choices |
| D12 | Served model `final_resnet18_s42`, chosen before the test run (logged in `experiments.log`) | best mean val macro-F1 by architecture (ResNet-18 0.488), best val seed |
| D13 | Naive ResNet-50 run skipped | 30,800 images/epoch on the 4 GB GPU would take hours; ResNet-18 already shows 0.999 |
| D14 | `channels_last` removed; label smoothing not used | measured 5-8x slower; smoothing + class weights collapses to ModerateDemented |
| D15 | `final_evaluation.py` crashed once while loading the XGBoost artifact after printing 14 test results; fixed the loader only and re-ran; the 14 repeated numbers are identical | full disclosure of a second test pass; no model/choice changed |
| D16 | torch 2.14.1 + cu130 wheels (RTX 3050, CUDA 13 driver), installed into the existing `.venv`; torch wheel downloaded with curl because pip stalled | GPU training; no change to existing packages (pip freeze diff = additions only) |
| D17 | Venv path is `_backend/alzheimerpredictor/.venv` (the task text said `_backend\.venv`, which does not exist) | `START_BACKEND.bat` uses this venv |
| D18 | Two earlier uncommitted backend fixes committed as the first commit on the branch | the branch must contain the working tabular API |
| D19 | Datasets ignored via `datasets/.gitignore` (`*`) because a parent folder that is itself a git repo would otherwise see them; backend `.gitignore` covers outputs/models/datasets | never commit data or models |
| D20 | No late fusion with the tabular model | no patient has both modalities; documented as future work |

## 8. Limitations
- **Patient-level leakage:** removed as far as the reconstructed IDs allow; repeat scans of one person under two IDs cannot be excluded (3.2% of held-out slices have a near-identical training slice).
- **Tiny classes:** ModerateDemented = 2 patients (1 for training, 1 for testing, none for validation). Its test numbers describe one person and must not be generalised.
- **Small test set:** 30 patients; 95% CI for macro-F1 ±~0.1 (subject-cluster bootstrap). Seeds vary by ±0.02–0.03.
- **Data:** public Kaggle data of unclear provenance (probably OASIS-derived), 2D axial JPEG slices (lossy, 8-bit), not 3D volumes; the label is a per-patient stage applied to every slice, so many slices carry no visible disease signal. Augmented, non-clinical copies were excluded from honest training.
- **No external or clinical validation.** Both folders come from the same source. No scanner, site, age or sex information is available to check confounding (e.g. ventricle size also grows with age).
- **Reduced setup:** single 4 GB laptop GPU; at most 25 epochs; small search (one seed); naive ResNet-50 not run. Numbers come from full data, not a subset, but the search is small.
- Grad-CAM is coarse (7x7) and qualitative.
- The API expects images that look like the training data (skull-stripped axial slice). Other images get an answer anyway; there is only a weak foreground check (`input_quality_warning`).

## 9. Next steps
1. **Ask Yasemin** (questions below), especially where her "extremely low" numbers came from.
2. For the paper, report the patient-level numbers (§0) as the result. Report the naive number only as a demonstration of leakage, together with the shuffled-label test.
3. If more MRI data is available, train with real patient IDs and 3D volumes (e.g. OASIS/ADNI with a data agreement). Consider merging Mild+Moderate or using a binary demented-vs-non-demented task, which this data can support better.
4. Repeat the hyper-parameter search with ≥3 seeds per setting (or patient-grouped CV) to reduce selection noise; consider patient-level aggregation (average over slices) as the reported unit.
5. Add an input-domain check to the API (e.g. reject images that are not skull-stripped axial slices) before any demo.
6. Late fusion with the tabular model only if a dataset with both modalities for the same patients becomes available.
7. Merge `feature/mri-integration` into `main` only after review. It does not delete `src/data/processed/*`, and `requirements.txt` is unchanged.

### Questions for Yasemin
1. Which folder exactly did you train on (`img_processed` = which Kaggle download, and what was the folder layout inside)?
2. Which script and flags gave the "extremely low" results, and what were the numbers (accuracy? F1? on which split)?
3. Did you ever run on GPU with `--num-workers > 0` (that crashes on Windows), or with `--rotation-correction` (crashes on numpy 2)?
4. Do you know the original source of the images (OASIS?) or any patient/scan IDs, ages, or scanner info?
5. Did you deliberately delete `src/data/processed/*` in your commit? (It breaks the tabular predictions; it was not taken over.)

## 10. Where things are
- Code: `src/imgProcessing/*.py`, `src/image_api.py`, mount block in `src/app.py`, `requirements-image.txt`, `tests/test_mri_unit.py`, `tests/test_mri_api.py`.
- Outputs (git-ignored): `src/imgProcessing/output/` with `inventory/`, `cache/`, `splits/` (`split_subject.csv`, `split_image.csv`, `split_naive.csv`, `groups.csv`), `models/` (classical, leakage, `dl_runs/<run>/` with `best.pt`, `last.pt`, `meta.json`, `history.csv`, `curves.png`, `train.log`; `served/`), `reports/` (`final_results.json/.csv`, `hp_search*.csv`, `classical_split_subject.json`, `grouping_report.json`, `splits_report.json`, `data_checks.json`, `gradcam/`, `fig_leakage_vs_honest.png`, `test_used.json`), `logs/`.
- Baselines and logs for the tabular check and bug re-verification: `_backend/mri_baseline/`.

## Türkçe kısa özet (düzeltme öncesi; güncel sonuçlar `docs/MRI_IMPROVEMENT_REPORT.md` içinde)

- **Çalışan kısım:** Yasemin'in MRI kodu `feature/mri-integration` dalına alındı ve hataları düzeltildi. Tablo modeline hiç dokunulmadı (13/13 test geçiyor, tahminler birebir aynı). API'ye `/predict-mri`, `/predict-mri/explain` (Grad-CAM) ve `/mri/info` eklendi. Model ilk istekte yükleniyor, model yoksa düzgün bir JSON hatası dönüyor.
- **Veri setleri:** `archive (1)` = uraninjo, `archive` = aryansinghal10. İkisi de aynı 6.400 orijinal görüntüden türemiş; `archive` birinciyi tamamen içeriyor. Bu yüzden bağımsız bir dış test seti yok. Hasta kimliklerini dosya adlarından çıkardık: 200 hasta × 32 kesit.
- **Dürüst ve naive sonuç:** aynı ResNet-18, naive bölmede **%99,9** doğruluk veriyor; hasta bazlı dürüst bölmede ise **dengeli doğruluk ~%51, macro-F1 ~0,48** (şans 0,25). Rastgele etiket testi sızıntıyı kanıtlıyor: hastalık bilgisi olmadan bile naive bölmede %44, dürüst bölmede %25,5 (şans seviyesi).
- **Riskler:** ModerateDemented'de yalnızca 2 hasta var. Test kümesi 30 hasta, bu yüzden her sayıda yaklaşık ±0,1 belirsizlik var. Veri 2D, Kaggle kaynaklı ve klinik doğrulaması yok. Model çoğunlukla ventrikül büyüklüğüne bakıyor; bu yaşla da artan bir özellik.
- **Kontrol etmen / Yasemin'e sorman gerekenler:** "çok düşük" sonucu hangi klasör ve hangi komutla aldığı (onun kodu bu veride naive bölmede 0,81 veriyor), görüntülerin asıl kaynağı ve hasta bilgisi, `src/data/processed/*` dosyalarını bilerek silip silmediği.
