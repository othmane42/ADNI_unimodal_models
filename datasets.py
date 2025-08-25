"""Loading dataset useful functions."""

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from seeder import seed_worker
from transformers import BatchFeature
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None


class CustomDataset(Dataset):
    def __init__(self,
                 df_X,
                 Y,
                 config):
        self.df_X = df_X.copy()
        self.Y = Y.copy()
        self.colname_image = config.get("image_column", None)
        self.categorical_columns = config.get("cat_cols_dummies", []) if config.get(
            "make_dummies", True) else config.get("cat_cols", [])
        self.continuous_columns = config.get("cont_cols", [])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        output = {}
        # Handle image data
        if self.colname_image is not None:
            output["image"] = self.df_X.iloc[index][self.colname_image]

        # Handle tabular data
        if self.categorical_columns or self.continuous_columns:
            if self.categorical_columns:
                cat_cols = self.categorical_columns
            if self.continuous_columns:
                cont_cols = self.continuous_columns
            output["tabular"] = self.df_X.iloc[index][cat_cols + cont_cols]

        # Handle label
        try:
            output["label"] = self.Y[index]
        except IndexError as e:
            raise IndexError(
                f"Index {index} is out of bounds for the dataset.") from e

        return output

    def __len__(self):
        return len(self.df_X)


class MMClassificationCollator:
    def __init__(self, image_processing=None, tabular_processing=None):
        self.image_processing = image_processing
        self.tabular_processing = tabular_processing

    def process_image(self, batch):
        if "image" not in batch[0]:
            return None
        transform_fun, params, data_type = (
            self.image_processing["transform_fun"],
            self.image_processing["params"],
            self.image_processing["data_type"]
        )

        if data_type == "image":
            images = [Image.open(x["image"]).convert('RGB') for x in batch]
        elif data_type == "mri":
            images = [torch.tensor(np.load(x["image"]),
                                   dtype=torch.float32) for x in batch]
        elif data_type in ["mgz", "nii.gz"]:
            images = [x["image"] for x in batch]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        return_value = transform_fun(
            images=images, **params) if params else transform_fun(images) if transform_fun else images
        if isinstance(return_value, list) and len(return_value) > 1:
            if isinstance(return_value[0], BatchFeature):
                return_value = BatchFeature({"pixel_values": torch.stack(
                    [x["pixel_values"].squeeze() for x in return_value])})
        return_value = torch.stack(return_value) if isinstance(
            return_value, list) else return_value

        return return_value

    def process_tabular(self, batch):
        if "tabular" not in batch[0]:
            return None
        if callable(self.tabular_processing):
            tab_inputs = [x["tabular"].values for x in batch]
            tab_inputs = pd.DataFrame(
                tab_inputs, columns=batch[0]["tabular"].index)
            return self.tabular_processing(tab_inputs)
        else:
            tab_inputs = [x["tabular"].values for x in batch]
            tab_array = np.array(tab_inputs, dtype=np.float32)
            return torch.tensor(tab_array, dtype=torch.float32)

    def __call__(self, batch):
        output = {}

        # Process image data
        if self.image_processing is not None:
            output["image"] = self.process_image(batch)

        # Process tabular data
        if self.tabular_processing is not None or self.tabular_processing is True:
            output["tabular"] = self.process_tabular(batch)

        # Process labels
        output["label"] = torch.tensor([x["label"] for x in batch])

        return output


def get_adni_dataset_new(data_path, cat_cols, cont_cols, target_column,
                         subject_column, seed, normalize=False, make_dummies=True,
                         patient_based_split=True, drop_duplicates=False, **kwargs):
    # Initialize the label encoder
    le = LabelEncoder()

    data_df = pd.read_csv(data_path)
    data_df[target_column] = le.fit_transform(data_df[target_column])
    data_df[target_column] = data_df[target_column].astype(int)

    if cat_cols is not None and make_dummies:
        df_dummies = pd.get_dummies(data_df[cat_cols], columns=cat_cols)
        data_df = pd.concat(
            [data_df.drop(columns=cat_cols), df_dummies], axis=1)

    if normalize:
        scaler = StandardScaler()
        # Get all columns except the ones to exclude
        cols_to_exclude = ["nii_path", "Subject",
                           "RID", "PHASE", "Group", "Image Data ID", "PTID"]
        cols_to_scale = [
            col for col in data_df.columns if col not in cols_to_exclude]
        data_df[cols_to_scale] = scaler.fit_transform(data_df[cols_to_scale])

    if patient_based_split:
        # Perform patient-based split
        subjects_df = data_df[[subject_column,
                               target_column]].drop_duplicates()

        # Split the data into train and test sets
        train_subjects_df, tmp_subjects_df = train_test_split(
            subjects_df, test_size=0.2, random_state=seed, stratify=subjects_df[target_column])

        # Further split the train set into train and validation sets
        test_subjects_df, val_subjects_df = train_test_split(
            tmp_subjects_df, test_size=0.5, random_state=seed, stratify=tmp_subjects_df[target_column])

        train_df = data_df[data_df[subject_column].isin(
            train_subjects_df[subject_column])].reset_index(drop=True)
        val_df = data_df[data_df[subject_column].isin(
            val_subjects_df[subject_column])].reset_index(drop=True)
        test_df = data_df[data_df[subject_column].isin(
            test_subjects_df[subject_column])].reset_index(drop=True)
    else:
        # Perform classical split without considering patient id
        train_df, tmp_df = train_test_split(
            data_df, test_size=0.2, random_state=seed, stratify=data_df[target_column])
        val_df, test_df = train_test_split(
            tmp_df, test_size=0.5, random_state=seed, stratify=tmp_df[target_column])

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    if drop_duplicates:
        train_df = train_df.drop_duplicates(
            subset=[subject_column]).reset_index(drop=True)
        val_df = val_df.drop_duplicates(
            subset=[subject_column]).reset_index(drop=True)
        test_df = test_df.drop_duplicates(
            subset=[subject_column]).reset_index(drop=True)

    # Reset indexes for train, validation and test sets
    X_train = train_df.drop(columns=[target_column])
    X_val = val_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])
    y_train = pd.Series(train_df[target_column]).reset_index(drop=True)
    y_val = pd.Series(val_df[target_column]).reset_index(drop=True)
    y_test = pd.Series(test_df[target_column]).reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, le.classes_, le


def data_loader(X, y, collate_fn, config,
                shuffle=True):

    dataset = CustomDataset(X, y, config.dataset)
    dataloader = None

    if config.model.weight_samples:
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y)
        sample_weights = compute_sample_weight(
            class_weight={c: v for c, v in enumerate(class_weights.tolist())}, y=y)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                sampler=sampler,
                                worker_init_fn=seed_worker,
                                num_workers=config.num_workers,
                                pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=shuffle,
                                worker_init_fn=seed_worker,
                                num_workers=config.num_workers,
                                pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)
    return dataloader
