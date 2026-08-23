from pathlib import Path

from datasets import load_dataset

RAW_PATH = Path("data/raw")
SPLITS = ["train", "test", "validation"]

def main():
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("dair-ai/emotion")

    for split in SPLITS:
        output_file = RAW_PATH / f"{split}-00000-of-00001.parquet"
        dataset[split].to_parquet(output_file)
        print(f"Wrote {output_file} with {len(dataset[split]):,} rows")

if __name__ == "__main__":
    main()
