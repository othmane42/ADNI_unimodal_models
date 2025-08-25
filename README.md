### ADNI Unimodal Models

This repo helps you load unimodal models (MRI image or tabular) for ADNI classification and embedding generation.

### Install

```bash
pip install -r requirements.txt
```

### Data

- Download data from https://alumniumonsac-my.sharepoint.com/:f:/g/personal/536736_umons_ac_be/EnGtBxv83axEo4lbAFj-9SUB78ajLJMlrY89KahXB77OOw?e=3miQr6
- Make sure to update root folder in `nii_path` column once you unzip the MRI folder
- For MRI, paths to volumes (e.g., NIfTI) are in the column `nii_path`.
- I prepared a yaml file with  all necessary config attributes in `ADNI_dataset.yaml` file.

### Quick start

Refer to the notebook model_inference where you can find 

### Components

- Encoders: `encoders/image_encoders.py` (e.g., `Resnet3D`, `DenseNet3D`, `VIT`, `ResNet50`), `encoders/tabular_encoders.py`.
- Classifier: `models/classifier.py` (`CustomClassifier`, lazy input-dim init).
- Dataset/collator: `datasets.py` (`CustomDataset`, `MMClassificationCollator`, `get_adni_dataset_new`, `data_loader`).
- All processing can be found in preprocessing folder seperated depending on the modality used.

### Notes

- For patient-aware split, use `patient_based_split=True` and provide `subject_column`.
- Random seed used so far for train test split was 42
