import argparse
import json
import os

from datasets import load_dataset


def prepare_dataset(output_path):
    print("Loading dataset siyanzhao/Openthoughts_math_30k_opsd...")
    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Converting to slime format and saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            # slime expects a 'prompt' key by default, or we can specify --input-key
            # We also include 'label' which is the ground truth solution
            item = {"prompt": example["problem"], "label": example["solution"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/opsd_math_30k.jsonl")
    args = parser.parse_args()
    prepare_dataset(args.output)
