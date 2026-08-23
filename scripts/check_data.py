from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw")
LABEL_NAMES = {
    0: "sadness", 
    1: "joy", 
    2: "love", 
    3: "anger", 
    4: "fear", 
    5: "surprise"
}
SPLITS = ["train", "test", "validation"]


def load_split(split: str) -> pd.DataFrame:
    files = list(RAW_PATH.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file found for {split}")
    return pd.read_parquet(files[0])


def check_split(split: str):
    df = load_split(split)
    print(f"Split : {split}")
    print(f"Rows : {len(df):,}")
    print(f"Columns : {list(df.columns)}")
    print("Label distribution for dataset:")
    counts = df["label"].value_counts().sort_index()
    for label_id, count in counts.items():
        name = LABEL_NAMES.get(label_id, "empty")
        print(f"{label_id} {name:<10} {count:>5}")


def main():
    for split in SPLITS:
        try:
            check_split(split)
        except FileNotFoundError as e:
            print(f"FileNotFound:{e}")


if __name__ == "__main__":
    main()
