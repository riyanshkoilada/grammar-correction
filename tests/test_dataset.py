from datasets import load_dataset
import sys

try:
    print("Attempting to load liweili/c4_200m (slice)...")
    dataset = load_dataset("liweili/c4_200m", split="train[:10]", streaming=False)
    print("Successfully loaded dataset!")
    print("First example:")
    print(next(iter(dataset)))
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)
