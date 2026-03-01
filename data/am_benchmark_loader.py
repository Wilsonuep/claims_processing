from pathlib import Path

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
am_benchmark = pd.read_json("hf://datasets/amu-cai/llmzszl-dataset/llmzszl-test.jsonl", lines=True)

output_path = Path(__file__).resolve().parent / "am_benchmark.csv"
am_benchmark.to_csv(output_path, index=False)
print(am_benchmark.head())