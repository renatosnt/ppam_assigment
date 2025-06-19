import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing and stripping whitespace.
    Extend this function with more cleaning (e.g., punctuation removal, lemmatization) as needed.
    """
    return text.lower().strip()


def main(
    raw_path: Path,
    processed_dir: Path,
    partitions_dir: Path,
    test_size: float,
    val_size: float,
    num_clients: int,
    random_state: int,
):
    # Load raw data
    df = pd.read_csv(raw_path, sep='\t', names=['label', 'message'])

    # Clean text
    df['message'] = df['message'].apply(clean_text)

    # Split into train+val and test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state,
    )

    # Further split train into train and val
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val['label'],
        random_state=random_state,
    )

    # Save processed central splits
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir / 'train.parquet', index=False)
    val.to_parquet(processed_dir / 'val.parquet', index=False)
    test.to_parquet(processed_dir / 'test.parquet', index=False)

    # Partition train across clients
    partitions_dir.mkdir(parents=True, exist_ok=True)
    # Stratify partitioning: group by label, then split each group
    for client_id in range(1, num_clients + 1):
        partitions_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle training data
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Compute approximate partition sizes
    part_sizes = [len(train) // num_clients] * num_clients
    for i in range(len(train) % num_clients):
        part_sizes[i] += 1

    # Assign partitions
    start = 0
    for i, size in enumerate(part_sizes, start=1):
        part = train.iloc[start:start + size]
        part.to_parquet(partitions_dir / f'client_{i:02d}.parquet', index=False)
        start += size

    print(f"Saved processed splits in {processed_dir}")
    print(f"Saved {num_clients} client partitions in {partitions_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare SMS Spam data for centralized and federated training.')
    parser.add_argument('--raw-path', type=Path, default=Path('data/raw/SMSSpamCollection.csv'), help='Path to raw SMS spam CSV file')
    parser.add_argument('--processed-dir', type=Path, default=Path('data/processed'), help='Directory for processed train/val/test splits')
    parser.add_argument('--partitions-dir', type=Path, default=Path('data/partitions'), help='Directory for client partition files')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to reserve for testing')
    parser.add_argument('--val-size', type=float, default=0.1, help='Fraction of train+val to reserve for validation')
    parser.add_argument('--num-clients', type=int, default=10, help='Number of federated clients to simulate')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        partitions_dir=args.partitions_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        num_clients=args.num_clients,
        random_state=args.random_state,
    )
