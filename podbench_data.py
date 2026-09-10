#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load the PodBench benchmark from the Hugging Face Hub.

The benchmark is distributed as a dataset on the Hub:

    https://huggingface.co/datasets/cnxu/PodBench

Each row exposes the three fields the pipeline consumes directly — ``id``,
``instruction`` and ``input_prompt`` — plus a ``meta`` object holding the
descriptive fields (source material, speaker spec, requirement dimensions,
languages, token counts and provenance).

``load_podbench()`` parses ``meta`` and merges it back, returning records
identical to the reference JSON released with the paper, so scores are
comparable regardless of how the benchmark was obtained.

Usage:

    from podbench_data import load_podbench

    data = load_podbench()
    print(len(data))                    # 800
    print(data[0]["input_prompt"])      # the fully assembled model input
    print(data[0]["dims"])              # {"脚本结构": "问题导向"}

To keep ``meta`` as a nested object instead of merging it:

    data = load_podbench(flatten=False)
    print(data[0]["meta"]["dims"])
"""

import json

DATASET_ID = "cnxu/PodBench"
SPLIT = "test"


def load_podbench(dataset_id=DATASET_ID, split=SPLIT, revision=None, flatten=True):
    """Return the benchmark as a list of dicts.

    With ``flatten=True`` (the default) the ``meta`` object is merged into each
    record, reproducing the layout of the reference JSON. With ``flatten=False``
    it is kept as a nested ``meta`` key.

    Requires ``datasets``:  pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading PodBench requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_id, split=split, revision=revision)

    records = []
    for row in ds:
        meta = row["meta"]
        if isinstance(meta, str):
            meta = json.loads(meta)

        record = {
            "id": row["id"],
            "instruction": row["instruction"],
            "input_prompt": row["input_prompt"],
        }
        if flatten:
            record.update(meta)
        else:
            record["meta"] = meta

        records.append(record)

    return records


def load_podbench_json(path):
    """Load the benchmark from a local reference JSON file instead of the Hub."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_podbench()
    print(f"Loaded {len(data)} queries from {DATASET_ID}")
    sample = data[0]
    print(f"  id: {sample['id']}")
    print(f"  dims: {sample['dims']}")
    print(f"  content passages: {len(sample['content'])}")
    print(f"  input_prompt chars: {len(sample['input_prompt'])}")
