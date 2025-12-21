import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
am_benchmark = pd.read_json("hf://datasets/amu-cai/llmzszl-dataset/llmzszl-test.jsonl", lines=True)
am_benchmark.to_csv("data/am_benchmark.csv", index=False)
print(am_benchmark.head())