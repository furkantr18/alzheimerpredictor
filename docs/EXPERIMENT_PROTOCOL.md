# Experiment protocol: improving honest (patient-level) MRI performance

Written and committed **before** any experiment of this phase (branch `feature/mri-improvement`, 2026-10-05).
Changes after this commit are added at the end under "Amendments" with a timestamp and a reason; amendments made after seeing outer-fold results are marked *exploratory* and excluded from headline claims.

## 1. Data and units
- 6,400 original axial slices (dataset copies are never used). Unit of analysis = **patient group**: 197 groups (200 reconstructed patients; 3 ambiguous pairs merged) with 32 or 64 slices.
- Labels are per patient and copied to every slice. Group counts: NonDemented 99, VeryMildDemented 69, MildDemented 27, ModerateDemented 2.
- The model input is pixels only: the shared preprocessing (`mri_preprocess.py`, 224 px cache). File names, folder names, class and patient IDs are never model inputs. Patient IDs define **bags and folds** only. Slice order (from file names) is used only in the clearly flagged 2.5D and slice-position ablations. Every patient has slices at all positions 0..31, so position alone carries no class information (checked in code before use).

## 2. Tasks
| Task | Classes | Role |
|---|---|---|
| T3 (main) | Non / VeryMild / Mild+Moderate | primary |
| T2 | Non vs demented (VeryMild+Mild+Moderate) | secondary |
| T4 | 4 original classes | secondary; Moderate = 2 patients, reported with caveat |
| T3-ord | T3 as ordinal (Non < VeryMild < Mild+Moderate) | loss variant (CORAL / Frank-Hall) |

## 3. Evaluation design (the old 30-patient test split is BURNED)
- The historical one-shot test results (`final_results.json`) are kept only as a historical baseline and are never used for any decision.
- **Outer:** repeated stratified patient-grouped 5-fold CV, **3 repeats** (seeds 101, 202, 303), stratified by the 4-class patient label, over all 197 groups. Every patient is predicted exactly once per repeat by a model that never saw it. The fold file is saved (`output/cv/folds.csv`) and shared by **every** method, so comparisons are paired.
- **Inner (inside each outer-training part only):** grouped stratified 3-fold CV for frozen-feature heads (hyper-parameters, aggregation); a grouped 80/20 inner train/val split for neural models (early stopping, epoch count, temperature). Ensemble members and weights are chosen on inner results only.
- Everything fitted is fitted on training folds only: scalers, PCA, calibrators/temperature, class weights, aggregation learners, stacking.
- **Headline method** = "auto-selected pipeline": in every outer fold, the inner CV picks one configuration among all pre-declared Stage A/B candidates. Its outer score is the honest estimate. Scores of individual configurations are reported as ablations; picking the best of them using outer results would be optimistic (selection bias), and this is stated in the report.

## 4. Metrics
- **Primary:** patient-level macro-F1 and balanced accuracy (T3).
- **Also:** patient-level quadratic weighted kappa (QWK) and within-one-stage accuracy (ordered tasks), ROC-AUC one-vs-rest (macro), slice-level macro-F1 and ROC-AUC, accuracy (for completeness; majority baseline shown), ECE for calibrated models.
- Patient-level prediction = aggregation of the patient's slice probabilities (default mean; alternatives in Stage A), or a MIL model.
- **Uncertainty:** metric per repeat → mean ± SD over the 3 repeats; 95% CI by patient-cluster bootstrap (2,000 resamples of the 197 groups, metric averaged over repeats inside each resample).
- **Paired comparison:** the same bootstrap resamples for both methods → 95% CI of the difference and the share of resamples ≤ 0. A difference whose CI contains 0 is reported as **not distinguishable**. With ~197 patients, differences around ±0.03 are expected to be noise.

## 5. Baseline (re-run on the same folds)
Fine-tuned ResNet-18 with the earlier recipe (224 px, batch 64, AdamW lr 3e-4 head / 3e-5 backbone, wd 1e-4, dropout 0.2, backbone frozen 3 epochs, augmentation strength 1, class-weighted CE, warm-up + cosine, ≤25 epochs, early stopping on inner-val loss with patience 6, AMP), task-specific head, for T3, T2 and T4, one seed per outer fold (fold seed), on all 15 outer folds. Patient level = mean of slice probabilities.

## 6. Candidate methods and search ranges (pre-declared)
**Stage A: frozen embeddings + heads.**
- Backbones: ResNet-50 (IMAGENET1K_V2), EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny (torchvision); DINOv2 ViT-S/14 and ViT-B/14 (torch.hub); BiomedCLIP (open_clip/HF) and RadImageNet only if downloadable without gated access (skipped and logged otherwise). Embedding = average of image and horizontally flipped image (flip TTA). Computed once (no labels involved).
- Pre-processing inside folds: StandardScaler → PCA(256, fit on training fold).
- Heads and grids (inner-CV selected):
  - LogReg L2: C ∈ {1e-3, 1e-2, 1e-1, 1}
  - LinearSVM: C ∈ {1e-4, 1e-3, 1e-2}, scores via softmax of decision values
  - Ridge: α ∈ {10, 100, 1000}
  - LightGBM: num_leaves ∈ {7, 15}, 200 trees, lr 0.05, feature_fraction 0.5
  - MLP (sklearn, 1 hidden layer 128, L2 alpha ∈ {1e-1, 1}, early stopping)
  - All heads use balanced class weights where supported.
- Aggregation (inner-selected): mean, median, 20% trimmed mean, top-8 mean, mean of log-probabilities, learned (LogReg on per-class mean/max/std of slice probabilities, fit on inner OOF slice probabilities).
- Ablation (flagged): + slice-position one-hot as an extra feature.

**Stage B: patient-level models.**
- Gated-attention MIL, mean-pooling MIL, max-pooling MIL on frozen embeddings (PCA 256 in fold). Grid (inner-val selected): hidden ∈ {64, 128}, dropout ∈ {0.25, 0.5}, slice-dropout ∈ {0, 0.3}, weight decay ∈ {1e-4, 1e-2}, lr 1e-3, ≤200 epochs, early stopping on inner-val loss (patience 20), 3 seeds averaged for the final model of each outer fold. Attention weights saved.
- 2.5D embeddings (neighbouring slices k−1, k, k+1 as RGB channels) for the backbone(s) chosen by inner CV, with the same heads and MIL (flagged: uses slice order).
- Handcrafted model: brain area, dark-pixel (CSF proxy) share, intensity mean/std/quantiles, central-region dark share (ventricle proxy), per slice → patient mean/max/std → LogReg.

**Stage C: fine-tuning (only if A/B beat the baseline on inner CV).** Partial unfreezing (last block), layer-wise LR decay, EMA, mixup, MRI-appropriate augmentation (rotation ≤10°, translation ≤5%, scale ±10%, intensity jitter, horizontal flip, no vertical flip). ≤2 configurations, T3 only, same folds.

**Stage D: targets/losses/calibration.** Class-weighted vs class-balanced sampling (MIL), focal loss (MIL, optional), ordinal CORAL (MIL) and Frank-Hall (LogReg) for T3-ord; temperature scaling on inner val; ECE and reliability plots. Never label smoothing with class weights.

**Stage E: ensembles.** Per outer fold, average the probabilities of the top-3 configurations by inner score (from different backbones/heads); stacking only with inner OOF predictions.

**Budget:** at most 40 configurations per method family (as counted in the registry); the final configuration of each neural family uses ≥3 seeds.

## 7. Sensitivity analysis
Stricter grouping: merge patient groups (union-find) whose slices have a cross-patient near-twin (64x64 correlation ≥ 0.98, same threshold as the earlier data check). Rebuild the folds on the merged groups and re-evaluate the headline method (and the baseline) once.

## 8. Logging
Every run (method, config, task, repeat, outer fold, seed, inner score, outer metrics, seconds) is a row in `output/cv/experiments_registry.csv`, appended automatically. The report states the number of configurations and runs.

## 9. Time estimate (RTX 3050 Laptop 4 GB, 16-thread CPU)
| Stage | Estimate |
|---|---|
| Folds, registry, metrics code | 0.5 h |
| Embeddings: 7 backbones x 6,400 slices x 2 (flip) | 0.3-0.5 h (+ downloads ~1 GB) |
| Stage A heads: 7 backbones x 5 heads x 15 folds, inner CV (CPU, parallel) | 0.5-1 h |
| Fine-tuned baseline: 3 tasks x 15 folds x ~3 min | 2-2.5 h (GPU, background) |
| Stage B MIL + 2.5D + handcrafted | 0.5-1 h |
| Stage C (if justified) | 1-2 h per configuration |
| Stage D/E, sensitivity, report | 1-1.5 h |

## Amendments
1. *2026-10-05, before any outer result.* Slice positions run 0..32 (33 values; every patient has 32 of them, one position missing) and the missing position depends on the class (ModerateDemented never has position 26; other classes miss 25 for part of the patients) because of the original file naming. Raw position would therefore leak a little class information. The flagged slice-position ablation uses **8 coarse bins (position // 4, capped at 7)**, which puts 25 and 26 in the same bin. 2.5D and MIL only use the *order* of slices, which carries no such information.
2. *2026-10-05, before any outer result.* The bootstrap uses a numpy re-implementation of the metrics (`cv_core.fast_metrics`), verified equal to scikit-learn on 900 random cases (max difference 4e-16), because scikit-learn made 2,000 resamples take ~2.5 min per method. Point estimates still use scikit-learn.
3. *2026-10-05, before any outer result.* BiomedCLIP needed `transformers` for its text tower (installed); RadImageNet skipped (no official ungated programmatic download for torch/timm). Handcrafted features are evaluated through the same nested pipeline as embeddings (`handcrafted.py` → `stage_a_heads.py --emb handcrafted`).
