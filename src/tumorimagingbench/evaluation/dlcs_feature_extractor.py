import pandas as pd
import os
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, data_csv_path):
    """Get DLCS dataset split."""
    df = pd.read_csv(data_csv_path)
    return df[df["benchmark_split"].str.startswith(split)]


def preprocess_row(row, image_base_path):
    """Preprocess a row from DLCS dataset."""
    row["image_path"] = f'{image_base_path}/{row["ct_nifti_file"]}'

    if os.path.exists(row["image_path"]):
        return row
    else:
        return None


def extract_features(output_path, data_csv_path, image_base_path, model_names=None):
    """
    Extract features for the DLCS dataset.

    Parameters:
    -----------
    output_path : str
        Where to save extracted features
    data_csv_path : str
        Path to CSV with dataset metadata
    image_base_path : str
        Base path where images are located
    model_names : list of str, optional
        Specific models to extract (if None, uses all available)
    """
    features = extract_all_features(partial(get_split_data, data_csv_path=data_csv_path), partial(preprocess_row, image_base_path=image_base_path), model_names=model_names)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features for the DLCS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using all available models
  python dlcs_feature_extractor.py --output features/dlcs.pkl

  # Extract using specific models
  python dlcs_feature_extractor.py --output features/dlcs.pkl \\
    --models CTFMExtractor CTClipVitExtractor

  # Extract with custom paths
  python dlcs_feature_extractor.py --output features/dlcs.pkl \\
    --data-csv /path/to/dlcs/data.csv \\
    --image-base /path/to/images
        """
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features/dlcs.pkl",
        help="Path where to save extracted features (default: features/dlcs.pkl)"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/home/suraj/Repositories/TumorImagingBench/data/eval/dlcs/data.csv",
        help="Path to DLCS metadata CSV file"
    )
    parser.add_argument(
        "--image-base",
        type=str,
        default="/mnt/data1/datasets/DukeLungNoduleDataset",
        help="Base path where DLCS images are located"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to extract (space-separated)"
    )

    args = parser.parse_args()
    extract_features(args.output, args.data_csv, args.image_base, args.models)
