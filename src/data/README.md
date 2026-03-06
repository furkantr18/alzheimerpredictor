# Data Folder Notes

This project currently uses an MRI dataset that is already preprocessed by the dataset author.

## Current MRI dataset status (`src/data/img_processed`)
- Source: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2
- Already skull-stripped (non-brain tissue removed)
- Already cleaned for training use
- Already augmented and upsampled for class balancing

## Class distribution
- `NonDemented`: 12,800 images
- `VeryMildDemented`: 11,200 images
- `MildDemented`: 10,000 images
- `ModerateDemented`: 10,000 images
- Total: 44,000 images

## About `src/imgProcessing/preprocess_mri.py`
Use this script only when you add a different MRI dataset that is not preprocessed yet.

Recommended flow for new raw data:
1. Put raw images in `src/data/img_unprocessed/` using class subfolders.
2. Run `python src/imgProcessing/preprocess_mri.py --input-dir src/data/img_unprocessed`.
3. Use generated outputs for model training only when needed.
