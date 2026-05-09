from base_process import make_mcq_process_fn
from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
ds = ds.map(make_mcq_process_fn("question", "options", "answer"), remove_columns=ds.column_names)
ds.to_json("data/json/mmlupro_test.jsonl")
