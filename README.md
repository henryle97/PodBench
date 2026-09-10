# PodBench: A Comprehensive Benchmark for Instruction-Aware Audio-Oriented Podcast Script Generation
<p align="center">
  📃 <a href="https://aclanthology.org/2026.acl-long.2019/" target="_blank">[Paper]</a> • 🔗 <a href="https://doi.org/10.18653/v1/2026.acl-long.2019" target="_blank">[DOI]</a> • 📊 <a href="https://huggingface.co/datasets/cnxu/PodBench" target="_blank">[Benchmark]</a>
</p>

<p align="center">
  <b>ACL 2026 (Long Papers), pages 43605–43621</b><br>
  Chenning Xu, Mao Zheng, Mingyu Zheng, Mingyang Song<br>
  <i>Large Language Model Department, Tencent</i>
</p>

## 📖 Overview

PodBench is a comprehensive benchmark for evaluating LLMs' ability to generate podcast scripts from real-world source material, spanning:
- **800 queries** (400 Chinese, 400 English)
- Inputs up to **21K tokens**
- **8 requirement dimensions**, including complex multi-speaker instructions
- **34 reference systems** evaluated in the paper, spanning proprietary and open-source models

Each query pairs a source document with a user instruction. Scripts are scored on two complementary axes — instruction following and audio-first script quality — combining quantitative constraint checking with LLM-based quality assessment.

This repository provides everything needed to **evaluate your own model on PodBench**: the full benchmark, the two judge rubrics, and scripts for generation, scoring, and leaderboard aggregation.

Our experiments reveal that while proprietary models generally excel, open-source models equipped with explicit reasoning demonstrate superior robustness in handling long contexts and multi-speaker coordination. Crucially, the analysis uncovers a persistent divergence where **high instruction following does not guarantee high content substance**.

## 🏗️ Benchmark Construction

PodBench is built by combining **model-augmented instruction synthesis** with **human expert verification**.

### 🤖 Instruction Synthesis

User instructions are synthesized with in-context prompting over a schema of eight requirement dimensions grounded in real podcast-creation requests:

- 🎙️ **Podcast Type** — e.g. interview, debate, narrative feature
- 🧩 **Script Structure** — e.g. general-specific-general, chronological
- 🌐 **Podcast Language** — Chinese or English
- 👤 **Speaker Profile** — personality, conversational style, domain expertise
- 👥 **Speaker Number** — solo, dialogue, or roundtable, not fixed at two
- 📏 **Podcast Length** — from ~1,000 to ~8,000 words
- 🎯 **Content Focus** — which parts of the source material to emphasize
- 🗣️ **Delivery Style** — tone and register for oral presentation

### ✍️ Human-in-the-Loop Filtering

From 2.6K initial instances, quality is ensured in two stages:
- **Safety and relevance review**: an LLM screens source documents for harmful content and judges instruction-document relevance
- **Expert verification**: 5 human experts re-check every document-instruction pair, removing low-quality instances and revising minor instruction errors

This yields the final set of 800 samples aligned with their input materials.

## 📈 Evaluation Framework

### Stage 1: Instruction Following (0–100)

The judge derives a sample-specific checklist from the user instruction, then verifies each item against the generated script. The score is the fraction of satisfied items, capturing whether explicit constraints such as language, speaker count, and target length are met.

### Stage 2: Podcast Script Quality (0–100)

An **audio-first** rubric that requires evidence-based critique with verbatim citations from the script. The evaluator first identifies the script type (monologue, dialogue, narrative feature) before scoring across three dimensions:

| Dimension | Points | Assesses |
|---|---|---|
| **Content Substance** | 45 | Unique insight and analytical depth beyond factual summarization |
| **Narrative Engagement** | 30 | Opening hooks, pacing, and transitions that sustain listener attention |
| **Conversational Naturalness** | 25 | Suitability for oral delivery; penalizes unspeakable syntax and jargon |

The final **Ave.** score is the mean of the two stages. All reported results use `anthropic/claude-opus-4.5` as the judge, which is the default in `evaluate_benchmark.py`.

## 🏆 Reference Results

Selected rows from Table 2 of the paper; see the paper for all 34 models.

| Model | Instruction Following | Content Quality (Overall) | Ave. |
|---|---|---|---|
| GPT-5.1 | 95.52 | **69.64** | **82.58** |
| Gemini-3-pro-preview | 96.09 | 63.85 | 79.97 |
| Claude-4-5-Sonnet | **96.58** | 63.25 | 79.91 |
| DeepSeek-R1-0528 | 94.27 | 61.15 | 77.71 |
| Qwen3-235B-A22B (thinking) | 94.32 | 60.42 | 77.37 |
| DeepSeek-V3-0324 | 92.71 | 59.61 | 76.16 |
| GPT-4o | 89.43 | 56.86 | 73.15 |
| LongWriter-zero-32B | 84.91 | 59.29 | 72.10 |
| Qwen2.5-7B-Instruct | 69.13 | 43.96 | 56.55 |

Two patterns recur across the leaderboard. First, satisfying explicit constraints does not translate into engaging dialogue: several systems score highly on instruction following while leaving substantial headroom on the quality rubric. Second, the largest deficit is consistently in **Content Substance** rather than surface form, indicating that sustaining grounded, non-trivial analysis across a long multi-speaker dialogue is considerably harder than maintaining fluency or structure.

## 🛠 Installation

```bash
git clone https://github.com/xucncn/PodBench.git
cd PodBench
pip install -r requirements.txt
```

For local generation with vLLM, additionally install:

```bash
pip install "vllm>=0.9.0" transformers
```

## 🤗 Benchmark Data

The benchmark lives on the Hugging Face Hub at
[`cnxu/PodBench`](https://huggingface.co/datasets/cnxu/PodBench) and is fetched
automatically by the scripts in this repository. To load it directly:

```python
from podbench_data import load_podbench

data = load_podbench()          # list of 800 dicts
print(data[0]["input_prompt"])  # the fully assembled model input
```

`load_podbench` merges the dataset's `meta` object back into each record. Using
`datasets` directly works too:

```python
from datasets import load_dataset

ds = load_dataset("cnxu/PodBench", split="test")
```

## 🚀 Quick Start

Evaluating a local model end to end — generate, score, then rank:

```bash
# 0. Point env.sh at your model (MODEL_PATH and MODEL_NAME)
vim inference/env.sh

# 1. Generate scripts for all 800 queries
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash inference/run_inference.sh

# 2. Score them (MODEL_NAME and the config string come from env.sh)
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
GEN_FILE=$(ls inference/llm_infer_results/<MODEL_NAME>/*/generations.jsonl)
python evaluate_benchmark.py \
    --model_file "${GEN_FILE}" \
    --output_file ./results/<MODEL_NAME>_eval.jsonl

# 3. Print the leaderboard
python calculate_scores.py --results_dir ./results --detailed
```

Each step is described in more detail below.

### 1. Generation

Generate scripts for the benchmark queries with your own pipeline, or use the provided vLLM scripts. Each query's fully assembled model input is in the `input_prompt` field.

Configure the model and decoding parameters in `inference/env.sh`:

```bash
MODEL_PATH="/path/to/your/model"   # HuggingFace model path or ID
MODEL_NAME="your-model-name"       # Display name for output dirs
TP_SIZE=1                          # GPUs per replica; raise if the model doesn't fit on one
IS_THINKING_MODE=0                 # 0=instruct, 1=thinking
MAX_NEW_TOKENS=12000
TEMPERATURE=1.0
TOP_P=0.8
TOP_K=40
```

Inference runs as a single process using vLLM data parallelism, so no manual sharding or merging is needed:

```bash
# One replica per GPU (dp = number of visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash inference/run_inference.sh

# Large model needing 2 GPUs per replica -> tp=2, dp=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=2 bash inference/run_inference.sh

# MoE model, sharding experts across replicas
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=2 ENABLE_EP=true bash inference/run_inference.sh
```

Generations are written to `inference/llm_infer_results/<MODEL_NAME>/<config>/generations.jsonl`.

If you generate with your own pipeline instead, write one record per query with the script in a `model_output` field. Both JSONL and a JSON array are accepted:

```bash
{"id": "...", "instruction": "...", "model_output": "..."}
```

### 2. Scoring

Evaluation uses [OpenRouter](https://openrouter.ai/) to reach the judge:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

```bash
python evaluate_benchmark.py \
  --model_file <generations.jsonl> \       # output of step 1
  --output_file ./results/<MODEL_NAME>_eval.jsonl \  # one file per model
  --max_workers 20                         # concurrent judge requests
```

Name the output `<model_name>_eval.jsonl`: `calculate_scores.py` takes the model label from the filename. Add `--is_thinking_model` for reasoning models so that only the final script is judged. Results are checkpointed, so an interrupted run resumes automatically.

### 3. Calculation

Aggregate the scoring results into a leaderboard. Every model with a `*_eval.jsonl` file in `--results_dir` becomes a row, so evaluate several models into the same directory to rank them together:

```bash
python calculate_scores.py \
  --results_dir ./results \          # directory of *_eval.jsonl files from step 2
  --detailed \                       # per-model rubric breakdown
  --export_csv ./scores.csv          # optional CSV export
```

The output reports Instruction Following, Podcast Script Quality with its three-dimension breakdown, and their average. Our per-sample judge outputs for the 34 systems in the paper will be released separately; the benchmark itself is available on the Hub at [`cnxu/PodBench`](https://huggingface.co/datasets/cnxu/PodBench).

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_file` | required | Model generations to evaluate |
| `--output_file` | required | Destination for judge outputs |
| `--evaluator_model` | `anthropic/claude-opus-4.5` | Judge model used in the paper |
| `--is_thinking_model` | off | Strip reasoning traces before judging |
| `--max_workers` | 20 | Concurrent judge requests |
| `--limit` | none | Evaluate only the first N samples |
| `--stages` | `stage2,stage3` | Rubrics to run |
| `--no_resume` | off | Ignore an existing checkpoint |

## 📄 License

Code is released under the MIT License. The benchmark data is intended for
non-commercial academic research, and source documents retain their original
licenses.

## 🤝 Contributing and Contact

PodBench aims to be a reliable and sustainable community resource for long-form, audio-centric script generation. If you are interested in leaderboard submissions or any further discussion, please reach out via GitHub issues.

## 📝 Citation

```bibtex
@inproceedings{xu-etal-2026-podbench,
    title = "{P}od{B}ench: A Comprehensive Benchmark for Instruction-Aware Audio-Oriented Podcast Script Generation",
    author = "Xu, Chenning  and
      Zheng, Mao  and
      Zheng, Mingyu  and
      Song, Mingyang",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2019/",
    doi = "10.18653/v1/2026.acl-long.2019",
    pages = "43605--43621",
    ISBN = "979-8-89176-390-6"
}
```
