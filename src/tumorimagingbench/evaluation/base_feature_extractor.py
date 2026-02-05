import sys
import torch
from tqdm import tqdm
import pickle
import multiprocessing

from tumorimagingbench.models import get_available_extractors, get_extractor

def get_model_list(model_names=None):
    """
    Return list of model classes to use for feature extraction.

    Parameters:
    -----------
    model_names : list of str, optional
        Specific model names to extract. If None, uses all available models.
        Example: ['CTFMExtractor', 'CTClipVitExtractor']

    Returns:
    --------
    list
        List of model classes ready for instantiation

    Notes:
    ------
    - Handles missing dependencies gracefully
    - Models are retrieved dynamically from the models registry
    """
    if model_names is None:
        model_names = get_available_extractors()

    model_classes = []
    for name in model_names:
        try:
            model_cls = get_extractor(name)
            model_classes.append(model_cls)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if not model_classes:
        raise RuntimeError("No valid models found")

    return model_classes


def extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn):
    """Extract features for a single model across all splits."""
    model = model_class()
    print(f"\nProcessing {model.__class__.__name__}")
    model.load()

    model_features = {}
    model = model.to("cuda")

    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_df = get_split_data_fn(split)
            if split_df is None:
                continue

            model_features[split] = []

            for _, row in tqdm(
                split_df.iterrows(), total=len(split_df)
            ):
                row = preprocess_row_fn(row)
                if row is None:
                    continue

                
                image = model.preprocess(row)
                image = image.unsqueeze(0)

                image = image.to("cuda")
                feature = model.forward(image)
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                model_features[split].append({
                    "feature": feature,
                    "row": row
                })

    return model_features


def extract_all_features(get_split_data_fn, preprocess_row_fn, model_names=None):
    """
    Extract features using available models.

    Parameters:
    -----------
    get_split_data_fn : callable
        Function that takes split name and returns DataFrame
    preprocess_row_fn : callable
        Function that preprocesses a single row
    model_names : list of str, optional
        Specific models to use. If None, uses all available models.

    Returns:
    --------
    dict
        Dictionary with model names as keys and features as values
    """
    feature_dict = {}
    model_classes = get_model_list(model_names=model_names)

    print(model_classes)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            extract_features_for_model,
            [(model_class, get_split_data_fn, preprocess_row_fn) for model_class in model_classes]
        )

    for model_class, model_features in zip(model_classes, results):
        feature_dict[model_class.__name__] = model_features

    return feature_dict


def save_features(feature_dict, output_file):
    """Save extracted features to a file."""
    with open(output_file, 'wb') as f:
        pickle.dump(feature_dict, f)
    print(f"Features saved to {output_file}")


if __name__ == "__main__":
    import pandas as pd

    def get_split_data(split):
        """Get dataset split."""
        split_paths = {
            "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/train.csv",
            "val": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/val.csv",
            "test": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/test.csv"
        }
        if split not in split_paths:
            raise ValueError(f"Invalid split: {split}")
        return pd.read_csv(split_paths[split])[:10]

    def preprocess_row(row):
        """Preprocess a row from the dataset."""
        return row

    def extract_features():
        """Extract features for the dataset."""
        features = extract_all_features(get_split_data, preprocess_row)
        save_features(features, 'features/test.pkl')

    extract_features()
