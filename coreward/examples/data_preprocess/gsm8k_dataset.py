import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "gsm8k"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, 'main', trust_remote_code=True)

    train_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    # instruction_following = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{}."
    # instruction_following = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. Please provide a step-by-step solution to the following problem."
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            solution = example.pop("answer")
            ans = solution.split('### ')[1].strip()
            print(ans)
            data = {
                "data_source": "DigitalLearningGmbH/MATH-lighteval",
                "prompt": [
                    {"role": "system", "content": instruction_following},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ans},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    print(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
