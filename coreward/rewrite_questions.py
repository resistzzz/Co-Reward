import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import copy

import os 
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def build_prompt_from_message(content, tokenizer, enable_thinking=False):
    instruction = """You are given a math problem. Please rewrite it using different wording and a different real-world scenario, while keeping the underlying mathematical meaning and answer exactly the same.

Guidelines:
1. Do not change the math logic or the final answer.
2. Use different words and a new context to make it look like a different problem.
3. Avoid copying phrases or sentence structures from the original.
4. Make sure the rewritten question is natural, clear, and solvable.
5. Output ONLY between the following markers, and strictly in this format (no extra explanation):

### RESULT_START
ORIGINAL:
<original question>
REWRITE:
<rewritten question>
### RESULT_END
"""
    full_prompt = instruction + "\n" + content.strip()
    messages = [{"role": "user", "content": full_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )


def extract_between_markers(text, start_marker="### RESULT_START", end_marker="### RESULT_END"):
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Failed to locate RESULT_START and RESULT_END markers.")
    block = text[start + len(start_marker):end].strip()
    # 按格式分割
    # 支持REWRITE块里再出现"ORIGINAL:"的情况，用split('REWRITE:', 1)
    parts = block.split('REWRITE:', 1)
    if len(parts) != 2:
        raise ValueError("Failed to split ORIGINAL and REWRITE in output.")
    original = parts[0].replace("ORIGINAL:", "").strip()
    rewritten = parts[1].strip()
    return original, rewritten


def main(args):
    # Load data
    df = pd.read_parquet(args.input_path)
    questions = df[args.question_column].tolist()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True
    )

    # Load model
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=8,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=20480
    )
    successful_indices = []
    # Prepare outputs
    batch_size = args.batch_size
    with open(args.output_jsonl, "w", encoding="utf-8") as jsonl_writer:

        for i in tqdm(range(0, len(questions), batch_size)):
            batch_indices = list(range(i, min(i + batch_size, len(questions))))
            batch_questions = [questions[idx] for idx in batch_indices]
            prompts = [
                build_prompt_from_message(q[1]['content'], tokenizer, args.enable_thinking)
                for q in batch_questions
            ]
            outputs = llm.generate(prompts, sampling_params)

            for j, output in enumerate(outputs):
                text = output.outputs[0].text
                try:
                    original, rewritten = extract_between_markers(text)
                    idx = batch_indices[j]
                    entry = copy.deepcopy(df.at[idx, args.question_column])
                    entry[1]['content'] = rewritten
                    df.at[idx, args.question_column] = entry   
                    successful_indices.append(idx)             
                    jsonl_writer.write(json.dumps({'original': original, 'rewritten': rewritten}, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[Warning] Failed to parse output:\n{text}\nReason: {e}")
                    continue

    # 只保留成功重写的样本
    if successful_indices:
        df = df.iloc[successful_indices].reset_index(drop=True)
        # 新增：保存对齐的原版parquet
        df_orig = pd.read_parquet(args.input_path)    # 重新读取原始数据，避免df被修改
        df_matched_orig = df_orig.iloc[successful_indices].reset_index(drop=True)
        df_matched_orig.to_parquet(args.output_original_parquet, index=False)
    else:
        print("Warning: No successful rewrites found! Output will be an empty DataFrame.")

    # Save final rewritten dataframe
    df.to_parquet(args.output_parquet, index=False)
    print(f"Finished! Saved to {args.output_parquet}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rewrite math problems using vLLM and Qwen3-32B.")
    parser.add_argument("--input_path", type=str, default="data/math/train.parquet")
    parser.add_argument("--output_jsonl", type=str, default="data/math/train_rewrite_Qwen3-32B.jsonl")
    parser.add_argument("--output_parquet", type=str, default="data/math/train_rewrite_Qwen3-32B.parquet")
    parser.add_argument("--output_original_parquet", type=str, default="data/math/train_original.parquet")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--question_column", type=str, default="prompt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3's thinking mode")
    args = parser.parse_args()
    main(args)
