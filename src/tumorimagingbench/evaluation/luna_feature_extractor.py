import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, train_csv, val_csv, test_csv):
    """Get dataset split."""
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the LUNA dataset."""
    return row


def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """
    Extract features for the LUNA dataset.

    Parameters:
    -----------
    output_path : str
        Where to save extracted features
    train_csv : str
        Path to training CSV
    val_csv : str
        Path to validation CSV
    test_csv : str
        Path to test CSV
    model_names : list of str, optional
        Specific models to extract (if None, uses all available)
    """
    features = extract_all_features(partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv), preprocess_row, model_names=model_names)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features for the LUNA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using all available models
  python luna_feature_extractor.py --output features/luna.pkl

  # Extract using specific models
  python luna_feature_extractor.py --output features/luna.pkl \\
    --models CTFMExtractor CTClipVitExtractor

  # Extract with custom paths
  python luna_feature_extractor.py --output features/luna.pkl \\
    --train-csv /path/to/train.csv \\
    --val-csv /path/to/val.csv \\
    --test-csv /path/to/test.csv
        """
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features/luna.pkl",
        help="Path where to save extracted features (default: features/luna.pkl)"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/luna16/train.csv",
        help="Path to training CSV"
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/luna16/val.csv",
        help="Path to validation CSV"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/luna16/test.csv",
        help="Path to test CSV"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to extract (space-separated)"
    )

    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
