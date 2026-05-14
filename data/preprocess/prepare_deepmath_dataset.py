import argparse
import json
import os
import random

from datasets import load_dataset


def prepare_dataset(output_path, difficulty_min, difficulty_max, sample_size, seed):
    print("Loading dataset zwhe99/DeepMath-103K...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    ds = ds.filter(lambda x: difficulty_min <= x["difficulty"] <= difficulty_max)
    print(f"After difficulty {difficulty_min}–{difficulty_max} filter: {len(ds)} examples")

    if sample_size is not None and sample_size < len(ds):
        random.seed(seed)
        indices = random.sample(range(len(ds)), sample_size)
        ds = ds.select(indices)
        print(f"Sampled {len(ds)} examples")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            item = {"prompt": example["question"], "label": example["final_answer"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. {len(ds)} examples written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepMath-103K to slime RL training format.")
    parser.add_argument("--output", default="data/json/deepmath_rl.jsonl")
    parser.add_argument("--difficulty-min", type=float, default=7.0)
    parser.add_argument("--difficulty-max", type=float, default=9.0)
    parser.add_argument("--sample-size", type=int, default=None, help="Cap on examples; omit to use all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.output, args.difficulty_min, args.difficulty_max, args.sample_size, args.seed)
