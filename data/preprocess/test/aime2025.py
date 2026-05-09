from base_process import make_math_process_fn
from datasets import load_dataset

ds = load_dataset("MathArena/aime_2025", split="train")
ds = ds.map(make_math_process_fn(), remove_columns=ds.column_names)
ds.to_json("data/json/aime2025_test.jsonl")
