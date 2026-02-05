"""
DummyDataset Feature Extractor

This module demonstrates how to create a feature extractor for a new dataset
in the TumorImagingBench framework. It serves as a template for adding support
for additional datasets.

The pattern involves:
1. Define create_get_split_data() - returns a function that loads dataset splits
2. Define preprocess_row() - prepares each row for model inference
3. Define extract_features() - main entry point that uses extract_all_features()
4. Add argparse configuration for flexible path handling

Dataset Structure Expected:
- CSV files with columns: image_path, coordX, coordY, coordZ, and optional label
- NIFTI images in the specified paths
- Train/val/test splits or single split as needed

Usage:
    # Command line - extract using all available models
    python dummy_dataset_feature_extractor.py --output features/dummy_dataset.pkl

    # Extract using specific models
    python dummy_dataset_feature_extractor.py --output features/dummy_dataset.pkl \\
      --models CTFMExtractor CTClipVitExtractor

    # Extract with custom paths
    python dummy_dataset_feature_extractor.py --output features/dummy_dataset.pkl \\
      --train-csv /path/to/train.csv \\
      --val-csv /path/to/val.csv \\
      --test-csv /path/to/test.csv

    # Programmatic usage
    from dummy_dataset_feature_extractor import extract_features
    extract_features('features/dummy_dataset.pkl')
"""

import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, train_csv, val_csv, test_csv):
    """Get dataset split by name."""
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }
    if split not in split_paths:
        return None
    try:
        return pd.read_csv(split_paths[split])
    except FileNotFoundError:
        print(f"Warning: {split} CSV not found at {split_paths[split]}")
        return None


def preprocess_row(row):
    """
    Preprocess a single row from the dataset.

    This function is called for each sample in get_split_data() and should:
    - Validate required columns exist
    - Check coordinates are valid
    - Optionally filter out invalid samples
    - Return row in format expected by model.preprocess()

    Parameters:
    -----------
    row : pandas.Series
        A single row from the dataset DataFrame

    Returns:
    --------
    dict or None
        Preprocessed row data or None to skip this sample

    Notes:
    ------
    - Returning None skips a sample (e.g., if image file missing)
    - The returned dict will be passed to model.preprocess()
    - Must include: image_path, coordX, coordY, coordZ
    - Can include additional metadata (label, patient_id, etc.)
    """
    # In this dummy implementation, we just pass through
    # In a real dataset, you might:
    # - Validate coordinates
    # - Check if image files exist
    # - Filter by clinical criteria
    # - Perform data cleaning

    return row


def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """
    Extract features for the Dummy Dataset.

    This is the main entry point for feature extraction. It:
    1. Calls extract_all_features() to run all models
    2. Saves results using save_features()

    Parameters:
    -----------
    output_path : str
        Where to save extracted features (pickle file)
    train_csv : str
        Path to training annotations CSV
    val_csv : str
        Path to validation annotations CSV
    test_csv : str
        Path to test annotations CSV
    model_names : list of str, optional
        Specific models to extract (if None, uses all available)
        Example: ['CTFMExtractor', 'CTClipVitExtractor']

    Returns:
    --------
    None
        Saves features to output_path

    Examples:
    ---------
    >>> # Extract all models
    >>> extract_features('features/dummy.pkl',
    ...                   'data/train.csv', 'data/val.csv', 'data/test.csv')
    >>>
    >>> # Extract specific models
    >>> extract_features('features/dummy.pkl',
    ...                   'data/train.csv', 'data/val.csv', 'data/test.csv',
    ...                   model_names=['CTFMExtractor'])
    """
    print("=" * 70)
    print("TumorImagingBench Feature Extraction - Dummy Dataset")
    print("=" * 70)

    # Extract features using all available models
    features = extract_all_features(partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv), preprocess_row, model_names=model_names)

    # Save features to disk
    save_features(features, output_path)

    print("=" * 70)
    print("✓ Feature extraction completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features for a custom/dummy dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  Extract using default paths (update these to your actual data paths):
  python dummy_dataset_feature_extractor.py --output features/dummy_dataset.pkl

EXAMPLES:
  # Extract all available models with default paths
  python dummy_dataset_feature_extractor.py --output features/dummy.pkl

  # Extract specific models only
  python dummy_dataset_feature_extractor.py --output features/dummy.pkl \\
    --models CTFMExtractor CTClipVitExtractor

  # Use custom data paths
  python dummy_dataset_feature_extractor.py --output features/dummy.pkl \\
    --train-csv /mnt/data/my_dataset/train.csv \\
    --val-csv /mnt/data/my_dataset/val.csv \\
    --test-csv /mnt/data/my_dataset/test.csv

  # List available models
  python -c "from models import get_available_extractors; print(get_available_extractors())"

CREATING NEW EXTRACTORS:
  1. Copy this file to your_dataset_feature_extractor.py
  2. Update create_get_split_data() for your data format
  3. Update preprocess_row() for any special preprocessing
  4. Update default paths in argparse
  5. Run: python your_dataset_feature_extractor.py --output features/your_dataset.pkl
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default="features/dummy_dataset.pkl",
        help="Path where to save extracted features (default: features/dummy_dataset.pkl)"
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/dummy_dataset/train.csv",
        help="Path to training annotations CSV"
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/dummy_dataset/val.csv",
        help="Path to validation annotations CSV"
    )

    parser.add_argument(
        "--test-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/dummy_dataset/test.csv",
        help="Path to test annotations CSV"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to extract (space-separated). If not specified, uses all available."
    )

    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
