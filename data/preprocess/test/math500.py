from base_process import MATH_SYSTEM, make_math_process_fn
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
ds = ds.map(make_math_process_fn(system_prompt=MATH_SYSTEM), remove_columns=ds.column_names)
ds.to_json("data/json/math500_test.jsonl")
