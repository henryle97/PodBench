#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate podcast scripts for the PodBench queries with vLLM.

A single process serves the whole benchmark: data parallelism replicates the model
across GPUs and vLLM schedules the prompts internally, so there is no need to shard
the input or merge per-GPU output files.

Queries are pulled from the Hugging Face Hub by default; pass ``--input-data-file``
to read a local JSON copy instead.

Requires vLLM >= 0.9.0 for offline data-parallel inference.
"""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from podbench_data import load_podbench, load_podbench_json  # noqa: E402

# Chat templates that a few released checkpoints need special handling for.
GLM4_LONGWRITER = "LongWriter-glm4-9b"
LLAMA31_LONGWRITER = "LongWriter-llama3.1-8b"
LLAMA31_BASE = "Llama-3.1-8B"


def build_prompt(item, tokenizer, model_path, enable_thinking):
    """Render one benchmark query into the model's expected prompt format."""
    input_text = item["input_prompt"]

    # Base and instruction-tuned LongWriter checkpoints ship without a usable
    # chat template, so their documented raw formats are applied directly.
    if LLAMA31_LONGWRITER in model_path:
        return f"[INST]{input_text}[/INST]"
    if model_path.endswith(LLAMA31_BASE) and "instruct" not in model_path.lower():
        return input_text + "\n\n"

    messages = []
    if item.get("system_prompt"):
        messages.append({"role": "system", "content": item["system_prompt"]})
    messages.append({"role": "user", "content": input_text})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )


def resolve_stop_token_ids(tokenizer, model_path):
    """Return extra stop tokens for checkpoints that do not stop on EOS alone."""
    if GLM4_LONGWRITER not in model_path:
        return None
    return [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|observation|>"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run PodBench inference with vLLM")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Local path or HuggingFace model ID")
    parser.add_argument("--input-data-file", type=str, default=None,
                        help="Local benchmark file (JSON). Defaults to loading "
                             "cnxu/PodBench from the Hugging Face Hub.")
    parser.add_argument("--output-data-file", type=str, required=True,
                        help="Destination JSONL file")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPUs per model replica; raise only if one GPU cannot hold the model")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of model replicas across GPUs")
    parser.add_argument("--enable-expert-parallel", action="store_true",
                        help="Shard MoE experts instead of replicating them")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override the context length derived from the model config")
    parser.add_argument("--enable-thinking", type=int, default=0,
                        help="1 to enable the chat template's thinking mode")
    parser.add_argument("--max-new-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.005)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    args = parser.parse_args()

    if args.input_data_file:
        print(f"Loading queries from {args.input_data_file}")
        test_data = load_podbench_json(args.input_data_file)
    else:
        print("Loading queries from the Hugging Face Hub (cnxu/PodBench)")
        test_data = load_podbench()
    print(f"Loaded {len(test_data)} queries")

    output_file = os.path.expanduser(args.output_data_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    engine_kwargs = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.enable_expert_parallel:
        engine_kwargs["enable_expert_parallel"] = True
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    print(f"Loading {args.model_path} "
          f"(tp={args.tensor_parallel_size}, dp={args.data_parallel_size})")
    llm = LLM(**engine_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        n=args.num_return_sequences,
        stop_token_ids=resolve_stop_token_ids(tokenizer, args.model_path),
    )
    print(f"Sampling params: {sampling_params}")

    prompts = [build_prompt(item, tokenizer, args.model_path, args.enable_thinking)
               for item in test_data]
    print("Example prompt after applying the chat template:")
    print(prompts[0])

    # All prompts are submitted at once so vLLM can batch and load-balance them
    # across replicas itself.
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    with open(output_file, "w", encoding="utf-8") as f:
        for item, output in zip(test_data, outputs):
            item["model_output"] = output.outputs[0].text
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference finished. Results written to {output_file}")


if __name__ == "__main__":
    main()
