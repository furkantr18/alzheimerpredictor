# Improving honest MRI performance: report

Branch `feature/mri-improvement` (local, not pushed). Protocol: `docs/EXPERIMENT_PROTOCOL.md`, committed before the experiments, plus 4 dated amendments. Runs: 2026-10-05 20:20 to 2026-10-06 ~03:30 (about 7 h wall clock) on an RTX 3050 Laptop (4 GB) and a 16-thread CPU. Every number below was **run and verified** on this machine unless marked otherwise.

## 0. Bottom line

1. **No method beats the baseline in a way the data can distinguish.** The baseline is a fine-tuned ResNet-18, re-run on the same folds.
   - Main task T3 (Non / VeryMild / Mild+Moderate): baseline patient-level macro-F1 is **0.518 [0.459, 0.575]**.
   - The honest, fully nested auto-selected pipeline reaches **0.492 [0.436, 0.544]**, a difference of −0.026, not distinguishable.
   - The inner-selected 3-model ensemble reaches 0.474, −0.044, also not distinguishable.
   - The best single configurations reach 0.533 (DINOv2-S 2.5D + LogReg) and 0.523 (BiomedCLIP + Ridge). These are picked by looking at outer results, they are only +0.005 to +0.015 above the baseline, and every CI contains 0.
   - The same holds for T2 (binary) and T4 (4 classes).
2. **A bug in the existing pipeline was found and fixed. This is the most important result of the phase.**
   - 1,441 of 6,400 slices (22.5%) had been attached to the wrong patient of the same class. This was patient-level leakage.
   - With the corrected IDs, frozen-feature results dropped by about 0.08 macro-F1. Before the fix they looked like a clear improvement (ensemble 0.606 vs baseline 0.550, "distinguishable"). After the fix the improvement disappeared.
   - All numbers here use the corrected IDs. The earlier integration numbers were also affected; an erratum has been added to them.
3. **The ceiling of this data at patient level is about 0.50–0.53 macro-F1** for 3 stages (balanced accuracy ~0.53, ROC-AUC ~0.74), and **~0.72–0.75 for demented vs non-demented**.
   - Eight handcrafted global features (CSF/dark-pixel share, brain area, intensity) reach 0.474. A large part of the learnable signal is therefore "how much dark CSF / ventricle is visible".
   - Fold assignment alone moves a configuration by ~0.04 (sensitivity run). Differences of ±0.03 between methods are noise.

## 1. Evaluation design (as pre-registered)

- **Data:** 6,400 original slices = **200 patients × 32 slices** (after the ID fix; 199 groups under the strict rule). Dataset copies are not used. Labels are per patient.
- **Outer CV:** stratified patient-grouped 5-fold × 3 repeats (seeds 101/202/303). Each patient is predicted once per repeat. The same fold file is used for every method, so comparisons are paired.
- **Inner CV:** inside each outer-training part, grouped 3-fold CV for heads and MIL, and a grouped 80/20 split for fine-tuning (early stopping). Scalers, PCA, class weights, aggregators, temperature and ensemble members are all fit on training folds only.
- **Uncertainty:** metric per repeat → mean ± SD; 95% CI from a patient-cluster bootstrap (2,000 resamples); paired bootstrap for differences against the baseline.
- **The old 30-patient test split was not used for anything** (it is "burned").
- **Tasks:**

  | Task | Classes | Patients per class |
  |---|---|---|
  | T3 (main) | Non / VeryMild / Mild+Moderate | 100 / 70 / 30 |
  | T2 | Non vs demented | 100 / 100 |
  | T4 | 4 classes | 100 / 70 / 28 / 2; Moderate = 2 patients, so the CIs are very wide |

## 2. The patient-ID bug (protocol amendment 4)

- **What was wrong:** the dataset has no patient IDs; they were reconstructed from file names. Two parts of the naming do **not** follow the subject order of the main `classDemN` rows:
  - the last, partial row (slice 25),
  - the `a (b)` tail files (slices 26–32), whose numbers partly follow a string sort of file names.
- **Why the old check missed it:** it compared positions k and k+1, and the head/tail boundary always skips a position (24→26 or 25→27).
- **How it was found:** while building the protocol's stricter-grouping sensitivity analysis.
  - 157 cross-patient slice pairs with r ≥ 0.98, **all within the same class** (chance probability ~0).
  - They occurred only at positions 24–27, never at the same position in both patients.
- **Fix:** `subject_ids.build_tail_remap()` re-links the row-25 slices and the tails by slice continuity (Hungarian assignment within length-compatible blocks), with a regression test.

| Check | Before fix | After fix |
|---|---|---|
| Slices per patient | 194×32, 3×33, 3×31 (merged into 197 groups) | **200 × 32** |
| Median correlation across the head/tail boundary | 0.904 (= the level between *different* patients) | **0.992** |
| Same-patient consecutive slices less similar than the other-patient 99th percentile | 4.2% | **0.03%** |
| Cross-patient near-twin slice pairs (r ≥ 0.98) | 157 | **1** |
| Effect on frozen-feature results (same pipeline) | ResNet-50 heads 0.57–0.59 | 0.48–0.52 |

## 3. Results (patient level, corrected IDs, identical folds)

### T3: main task

Full table with every configuration: `output/cv/report/tables.md`.

| Method | Patient macro-F1 [95% CI] | Balanced acc. | QWK | ROC-AUC | Slice macro-F1 | Δ vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| **Baseline: fine-tuned ResNet-18** | **0.518 [0.459, 0.575]** | 0.539 | 0.428 | 0.735 | 0.477 | – |
| **Headline: nested auto-selection (A+B)** | 0.492 [0.436, 0.544] | 0.509 | 0.430 | 0.724 | – | −0.026, n.s. |
| Inner-selected ensemble of 3 (A+B) | 0.474 [0.415, 0.528] | 0.488 | 0.440 | 0.738 | – | −0.044, n.s. |
| Best single (outer-picked, optimistic): DINOv2-S 2.5D + LogReg | 0.533 [0.473, 0.586] | 0.571 | 0.469 | 0.739 | 0.473 | +0.015, n.s. |
| BiomedCLIP + Ridge | 0.523 [0.462, 0.582] | 0.567 | 0.460 | 0.752 | 0.486 | +0.005, n.s. |
| ResNet-50 + LightGBM | 0.521 [0.455, 0.580] | 0.526 | 0.480 | 0.755 | 0.480 | +0.003, n.s. |
| MIL mean pooling, ResNet-50 | 0.514 [0.447, 0.579] | 0.533 | 0.458 | 0.738 | – | −0.004, n.s. |
| MIL gated attention, BiomedCLIP | 0.504 [0.444, 0.565] | 0.533 | 0.422 | 0.743 | – | −0.013, n.s. |
| MIL CORAL ordinal (ResNet-50, mean) | 0.503 [0.437, 0.565] | 0.515 | 0.442 | 0.743 | – | −0.014, n.s. |
| Stage C: EfficientNet-B0 partial fine-tune + EMA + mixup | 0.492 [0.435, 0.546] | 0.529 | 0.440 | 0.730 | 0.474 | −0.026 [−0.074, +0.020], n.s. |
| Frank-Hall ordinal LogReg (ConvNeXt-T) | 0.475 [0.413, 0.532] | 0.518 | 0.415 | 0.720 | 0.461 | −0.043, n.s. |
| **Handcrafted 8 global features + LogReg** | 0.474 [0.409, 0.536] | 0.533 | 0.434 | 0.717 | 0.467 | −0.043, n.s. |
| Worst frozen (BiomedCLIP + MLP) | 0.449 [0.394, 0.502] | 0.460 | 0.382 | 0.701 | 0.451 | **−0.069, worse** |

Distinguishably **worse** than the baseline on T3: ConvNeXt-T Ridge, DINOv2-B Ridge, DINOv2-B MLP, BiomedCLIP MLP. **No configuration is distinguishably better** on any task. All other differences are not distinguishable from 0.

### T2 (demented vs non-demented) and T4 (4 classes)

| Task | Baseline | Auto-selection (A) | Ensemble of 3 (A) | Best single (outer-picked) | Distinguishably better |
|---|---|---|---|---|---|
| T2 | **0.718 [0.662, 0.768]** (bal-acc 0.718, AUC 0.774) | 0.707 [0.651, 0.758] | 0.714 [0.655, 0.770] | EfficientNet-B0 + MLP 0.745 (+0.027 [−0.015, +0.072]) | none (1 worse) |
| T4 | **0.377 [0.332, 0.529]** (bal-acc 0.411, AUC 0.710) | 0.360 | 0.364 | DINOv2-S + LogReg 0.376 | none (6 worse) |

T4 macro-F1 includes the ModerateDemented class with 2 patients, which no method gets right reliably. The T4 numbers mainly show that 4-class staging is not supported by this data.

### Ablations (T3, patient macro-F1)

- **Backbone** (mean inner score over 5 heads): ResNet-50 0.524 ≥ BiomedCLIP 0.518 ≥ EfficientNet-B0 0.516 ≥ ConvNeXt-T 0.513 ≥ DINOv2-S 0.512 ≥ EfficientNet-B3 0.503 ≥ DINOv2-B 0.493. The medical-domain BiomedCLIP is not better than ImageNet models, and the larger DINOv2-B is the worst.
- **Head:** strongly regularised linear heads (LogReg, LinearSVM, Ridge) and small LightGBM are about equal. MLP heads reach train macro-F1 1.00 and are among the worst outer scores.
- **Aggregation:** chosen per fold on inner CV, over 660 fold choices (44 configurations × 15 folds): learned aggregator 29%, top-8 mean 17%, mean 15%, log-mean 14%, median 13%, trimmed mean 12% (`results/A_*__T3.json` → `chosen`). No aggregation clearly wins; inner differences are small.
- **2.5D** (neighbouring slices as RGB): 0.468–0.533 vs 0.48–0.51 for 2D with the same head. Mixed, no consistent gain.
- **Slice position** (coarse 8 bins, flagged ablation): ConvNeXt-T LogReg 0.507 → 0.510, no gain.
- **MIL vs simple averaging:** gated attention 0.491/0.504, mean 0.514/0.507, max 0.470/0.497 (ResNet-50/BiomedCLIP). MIL does not beat averaging the slice probabilities.
- **Losses:**
  - Ordinal losses (CORAL 0.503, Frank-Hall 0.475) give no gain over cross-entropy.
  - Class-weighted CE was used everywhere.
  - Label smoothing was not combined with class weights (known trap).
  - Focal loss and class-balanced sampling were not run (optional in the protocol; skipped because MIL as a whole showed no gain).
- **Resolution:** all runs at 224 px. Image size was tuned in the earlier integration phase (160 px was worse) and not re-tuned here.
- **Fine-tuning variants:** full fine-tuning (baseline, 0.518) ≥ partial fine-tuning with EMA + mixup (0.492). Stage C was run with one configuration only (see Decisions).

## 4. Overfitting

| Model | Train patient macro-F1 | Inner-val | Outer test | Note |
|---|---|---|---|---|
| Fine-tuned ResNet-18 (baseline) | 0.62 | 0.52 | 0.50 (fold mean) | best epoch 1–5 (median 4); inner-val loss rises right after |
| Stage C (partial FT + EMA + mixup) | 0.75 | 0.49 | 0.49 | best epoch 6–17 |
| ResNet-50 + LightGBM | 0.97 | 0.52 | 0.52 | trees memorise slices; patient averaging limits the damage |
| Linear heads (LogReg/Ridge) | 0.74–0.84 | ~0.52 | 0.49–0.52 | |
| MLP heads | 1.00 | – | 0.45–0.51 | worst |
| MIL (attention/mean/max) | 0.75–0.91 | 0.47–0.53 | 0.47–0.51 | |
| Handcrafted 8 features | 0.51 | 0.49 | 0.47 | no overfitting, low ceiling |

- Curves for every fold: `output/cv/report/fig_curves_B_ft_resnet18_T3.png` (and `..._C_ft_effb0_...`).
- More capacity raises training scores without raising outer scores. The limit is the number of patients (~160 per training fold) and the weak per-slice signal, not model capacity or regularisation.

## 5. Calibration

| Model | Patient-level ECE (pooled over all held-out patients, mean of 3 repeats) |
|---|---|
| Baseline (uncalibrated) | 0.077 |
| Inner-selected ensemble of 3 | 0.088 |
| MIL gated attention + temperature scaling (fit on inner out-of-fold logits): ResNet-50 | 0.077 |
| Same, BiomedCLIP | 0.070 |

- Reliability plot: `output/cv/report/fig_reliability_T3.png`.
- Per-fold ECE (about 40 patients each) is about 0.15 because of small-sample noise. The pooled values are the better estimate.
- Temperature scaling did not make MIL noticeably better calibrated than the uncalibrated baseline.

## 6. MIL attention (which slices matter)

Figures: `output/cv/report/fig_attention_B_mil_attn_{biomedclip,resnet50}_T3.png`. Tables: `attention_by_position_*.csv`. Attention is shown relative to uniform (1.0), averaged over held-out patients.

- **Anatomy of the 32 positions:** position 0 is inferior and 32 is superior (`slice_positions_example.png`). Position 0 is at the level of the basal ganglia, the third ventricle and the temporal horns; position 32 is at the bodies of the lateral ventricles. All 32 slices come from a narrow slab around the ventricles.
- **BiomedCLIP MIL:** attention decreases steadily from inferior to superior. Positions 0–4 get 1.40–1.50× uniform, positions 28–32 get 0.64–0.71×. The model favours the lower slices, closest to the medial temporal lobe.
- **ResNet-50 MIL:** attention is almost uniform (0.98–1.01× at positions 0–4).
- **Interpretation:** the two backbones disagree, and attention MIL does not beat mean pooling (§3). The profile is descriptive only. It is not evidence about which anatomy drives the prediction.

## 7. Sensitivity analysis (stricter grouping)

| Run | Patient macro-F1 |
|---|---|
| Strict grouping (merge patients with any r ≥ 0.98 slice pair): baseline | 0.521 [0.465, 0.576] |
| Same, normal grouping | 0.518 |
| Strict grouping: most-selected candidate (ResNet-50 2.5D + LogReg) | 0.528 [0.464, 0.590]; paired vs strict baseline +0.007 [−0.045, +0.061], n.s. |
| Same candidate, normal grouping | 0.486 |

- After the ID fix the strict rule merges only 1 pair (199 groups).
- Results are stable for the baseline (0.518 → 0.521).
- The candidate's +0.04 change comes almost entirely from the **different random fold assignment**, since the groups are almost identical. This is a direct measurement of fold-assignment noise.
- Only the baseline and the most-selected candidate were re-run (not the whole candidate pool), because 2 merged groups did not justify ~6 h of recomputation. The protocol asked for "the best method"; the headline auto-selection has fold-varying members, so its most frequently selected member was used.

## 8. Number of configurations and compute

- **Corrected-ID runs:** 126 method/task configurations, 1,890 outer-fold runs. The inner grids come on top:
  - Stage A heads: 2–4 grid points each × 6 aggregations;
  - MIL: 16 configurations per model;
  - Stage A T3: 44 configurations = 35 backbone×head + handcrafted + ordinal + position + 6×2.5D;
  - plus 7 MIL, 1 Stage C, 3 baseline and 2 sensitivity runs.
- **Archived pre-fix runs (not used):** 834 outer-fold runs of 49 methods (`output/cv_v1_wrong_tail_ids/`).
- **Compute:**
  - per-fold run time summed over all runs: ~35 h of process time (CPU and GPU jobs ran in parallel);
  - wall clock ~7 h;
  - downloads ~1.3 GB of weights;
  - new packages: open_clip_torch, transformers (+ timm, huggingface_hub), all in `requirements-image.txt`.

## 9. Claims

**Can be claimed:**
- With leakage-free, patient-grouped, nested CV on this public dataset (200 patients, 2D axial JPEG slices), a fine-tuned ResNet-18 reaches patient-level macro-F1 0.52 [0.46, 0.58] for 3 stages and 0.72 [0.66, 0.77] for demented vs non-demented.
- Frozen foundation/medical backbones, MIL, 2.5D input, ordinal losses, partial fine-tuning and ensembles did **not** improve on this (126 configurations, no distinguishable gain).
- A reconstruction error of patient IDs (22.5% of slices) inflated earlier results by ~0.08 macro-F1. Patient-ID quality must be verified, not assumed.
- The naive split gives 99.9% (earlier phase), showing how much leakage dominates published results on this dataset.

**Must NOT be claimed:**
- Any "best single configuration" number as an achieved result: it was picked on outer results.
- Any improvement from foundation models, MIL or ensembling.
- Clinical usefulness, 4-stage staging ability, or ModerateDemented performance (2 patients).
- Generalisation beyond this dataset (no external data, no demographics, possible age confound through ventricle size).
- The old one-shot test numbers or the pre-fix CV numbers.

## 10. Limitations

- Only 200 patients, and only 2 ModerateDemented patients.
- Patient IDs are reconstructed; they are now verified on pixels (every patient 32 continuous slices), but they are not ground truth.
- No age, sex, scanner or site information, so the ventricle-size/age confound cannot be checked.
- 2D 8-bit JPEG axial slices of unknown preprocessing (probably OASIS-derived), not 3D volumes.
- The label is per patient and copied to all slices.
- No external validation.
- All runs at 224 px; one Stage C configuration; focal loss and class-balanced sampling not run. The fine-tuned baseline and Stage C use one seed per outer fold (15 different seeds overall), not 3 seeds per fold as the protocol asked for neural families; MIL used 3 seeds per fold. The optional refreshed Grad-CAM was not produced in this phase.

## 11. Serving

**Unchanged.** No method is a clear, CI-supported improvement over the baseline, so the served model and the endpoints were left as they were (protocol deliverable 3).

## 12. Recommended next steps

1. Real patient IDs and **more patients**, e.g. OASIS-1/OASIS-3 or ADNI with a data-use agreement. With ~200 patients, ±0.03 differences cannot be resolved.
2. **3D volumes** (full T1 scans) instead of 2D JPEG slices; standard preprocessing (skull-strip, registration); region measures (hippocampal volume, ventricle volume) as strong baselines.
3. Add **age and sex** to control the ventricle-size confound, and report against an age-only baseline.
4. Report T2 (binary) as the primary task for this data and treat 3/4-stage staging as exploratory.
5. Keep the nested, patient-grouped protocol and the pixel-level ID verification for any new dataset.
