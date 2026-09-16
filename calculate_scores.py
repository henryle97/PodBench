#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate PodBench evaluation results into a leaderboard.

Reads the per-sample judge outputs written by ``evaluate_benchmark.py`` and reports,
for each model, the Instruction Following score, the Podcast Script Quality score
with its per-dimension breakdown, and their average.

Note on naming: the paper's "Stage 1" (Instruction Following) corresponds to
``stage2`` in the code and result files, and the paper's "Stage 2" (Podcast Script
Quality) corresponds to ``stage3``. The offset is historical.
"""

import json
import os
import argparse
from collections import defaultdict
import glob

from podbench_rubric import DIMENSIONS, dimension_scores

# Quality rubric dimensions are named by the judge in the language of the prompt
# it was given, so their keys live in podbench_rubric. Aggregation here is keyed
# by the paper's English name, which makes a results directory holding both
# Chinese-judged and English-judged models add up into one table.


def load_eval_results(eval_file):
    """Load one model's results as records with ``stage2_result``/``stage3_result``.

    Two on-disk layouts are accepted: a legacy layout with both stage results on
    one line, and the incremental layout written by the evaluation script, which
    emits one line per (sample, stage). Records are merged by sample index and
    deduplicated in favour of valid scores, so retried lines are not
    double-counted in the averages.
    """
    merged = {}

    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue

            index = data.get('index')
            entry = merged.setdefault(index, {'index': index, 'id': None,
                                              'stage2_result': None, 'stage3_result': None})
            if entry['id'] is None:
                entry['id'] = data.get('id')

            stage = data.get('stage')
            if stage in ('stage2', 'stage3'):
                _merge_stage(entry, stage, data.get('eval_result'))
            else:
                for stage_name in ('stage2', 'stage3'):
                    _merge_stage(entry, stage_name, data.get(f'{stage_name}_result'))

    return list(merged.values())


def _merge_stage(entry, stage, value):
    """Store ``value`` unless it would overwrite an already-valid score."""
    if value is None:
        return
    key = f'{stage}_result'
    if entry[key] is None or not _is_valid_stage(stage, entry[key]):
        entry[key] = value


def _is_valid_stage(stage, result):
    """Whether a stage result carries a usable score rather than a failure marker."""
    if not result:
        return False
    if stage == 'stage2':
        return result.get('normalized_score', -1) >= 0
    return result.get('total_score', -1) >= 0


def _parse_dim_scores(stage3):
    """Return ``(pairs, complete)`` for the three quality dimensions.

    ``pairs`` holds ``(score, max_score)`` per dimension, in rubric order;
    ``complete`` is False if any dimension is missing or malformed. Dimension
    keys are resolved in whichever language the judge answered in.
    """
    return dimension_scores(stage3)


def analyze_single_model(eval_file, model_name):
    """Compute aggregate statistics for one model's evaluation results.

    Every metric is averaged over the same set of samples: those with a valid
    instruction-following score and a complete quality rubric. Using one shared
    denominator keeps the two axes, their average, and the per-dimension
    breakdown mutually consistent, which is the convention used in the paper.
    """
    results = load_eval_results(eval_file)

    stage2_scores = []
    stage2_checklist_scores = defaultdict(list)
    stage3_scores = []
    stage3_dim_scores = defaultdict(list)
    combined_scores = []

    # Which stages this file actually holds. `--stages` can run one rubric alone,
    # and a stage that was never run must not delete the other's data: the
    # complete-cases rule below is what the paper's numbers are computed under,
    # and it only applies between stages that were both attempted.
    ran_stage2 = any(item.get('stage2_result') for item in results)
    ran_stage3 = any(item.get('stage3_result') for item in results)

    for item in results:
        stage2 = item.get('stage2_result') or {}
        stage3 = item.get('stage3_result') or {}

        s2_score = stage2.get('normalized_score', -1)
        s3_score = stage3.get('total_score', -1)
        dim_pairs, complete = _parse_dim_scores(stage3) if stage3 else ([], False)

        s2_ok = s2_score >= 0
        s3_ok = s3_score >= 0 and complete

        # Complete cases only, when both rubrics were run.
        if ran_stage2 and ran_stage3 and not (s2_ok and s3_ok):
            continue

        if s2_ok:
            stage2_scores.append(s2_score)
            for check_item in stage2.get('checklist', []):
                stage2_checklist_scores['all'].append(check_item.get('score', 0))

        if s3_ok:
            stage3_scores.append(s3_score)
            for dimension, pair in zip(DIMENSIONS, dim_pairs):
                stage3_dim_scores[dimension.name].append(pair)

        # Each axis contributes 50 points so the average is out of 100. A
        # single-stage run has no average -- half a score reported out of 100
        # would rank it against two-stage runs as though it had scored zero on
        # the rubric nobody ran.
        if s2_ok and s3_ok:
            combined_scores.append(s2_score * 50 + s3_score * 0.5)

    stats = {
        'model_name': model_name,
        'total_samples': len(results),
        'stage2': {
            'valid_count': len(stage2_scores),
            'mean': sum(stage2_scores) / len(stage2_scores) if stage2_scores else 0,
            'min': min(stage2_scores) if stage2_scores else 0,
            'max': max(stage2_scores) if stage2_scores else 0,
        },
        'stage3': {
            'valid_count': len(stage3_scores),
            'mean': sum(stage3_scores) / len(stage3_scores) if stage3_scores else 0,
            'min': min(stage3_scores) if stage3_scores else 0,
            'max': max(stage3_scores) if stage3_scores else 0,
        },
        'stage3_dims': {}
    }

    for dim_key, scores in stage3_dim_scores.items():
        if not scores:
            continue
        dim_scores = [s[0] for s in scores]
        max_possible = scores[0][1]
        mean = sum(dim_scores) / len(dim_scores)
        stats['stage3_dims'][dim_key] = {
            'mean': mean,
            'max_possible': max_possible,
            'normalized_mean': mean / max_possible if max_possible > 0 else 0,
        }

    stats['combined'] = {
        'valid_count': len(combined_scores),
        'mean': sum(combined_scores) / len(combined_scores) if combined_scores else 0,
        'min': min(combined_scores) if combined_scores else 0,
        'max': max(combined_scores) if combined_scores else 0,
    }

    return stats


def print_leaderboard(all_stats, sort_by='combined'):
    """Print the leaderboard, ranked by the requested metric."""
    sort_keys = {
        'combined': lambda x: x['combined']['mean'],
        'stage2': lambda x: x['stage2']['mean'],
        'stage3': lambda x: x['stage3']['mean'],
    }
    if sort_by in sort_keys:
        sorted_stats = sorted(all_stats, key=sort_keys[sort_by], reverse=True)
    else:
        sorted_stats = all_stats

    print("\n" + "=" * 100)
    print(f"PodBench Leaderboard (sorted by {sort_by})")
    print("=" * 100)
    print(f"{'Rank':<5} {'Model':<35} {'InstFollow':<12} {'Quality':<12} {'Ave.':<12} {'Samples':<8}")
    print("-" * 100)

    for rank, stats in enumerate(sorted_stats, 1):
        print(f"{rank:<5} {stats['model_name'][:33]:<35} "
              f"{stats['stage2']['mean'] * 100:.2f}       "
              f"{stats['stage3']['mean']:.2f}/100   "
              f"{stats['combined']['mean']:.2f}/100  "
              f"{stats['combined']['valid_count']:<8}")

    print("=" * 100)
    print()


def print_detailed_stats(all_stats):
    """Print per-model statistics, including the quality rubric breakdown."""
    print("\n" + "=" * 100)
    print("Detailed Statistics by Model")
    print("=" * 100)

    for stats in all_stats:
        print(f"\n--- {stats['model_name']} ---")
        print(f"Total samples: {stats['total_samples']}")
        print()

        print("Instruction Following (paper Stage 1):")
        if stats['stage2']['valid_count']:
            print(f"  Valid: {stats['stage2']['valid_count']}")
            print(f"  Mean: {stats['stage2']['mean'] * 100:.2f}/100")
            print(f"  Range: [{stats['stage2']['min'] * 100:.2f}, "
                  f"{stats['stage2']['max'] * 100:.2f}]")
        else:
            print("  not run")
        print()

        print("Podcast Script Quality (paper Stage 2):")
        if stats['stage3']['valid_count']:
            print(f"  Valid: {stats['stage3']['valid_count']}")
            print(f"  Mean: {stats['stage3']['mean']:.2f}/100")
            print(f"  Range: [{stats['stage3']['min']:.2f}, {stats['stage3']['max']:.2f}]")
        else:
            print("  not run")

        if stats['stage3_dims']:
            print("  By dimension:")
            for dim, dim_stats in stats['stage3_dims'].items():
                name = dim
                print(f"    {name}: {dim_stats['mean']:.2f}/{dim_stats['max_possible']:.0f} "
                      f"({dim_stats['normalized_mean'] * 100:.1f}%)")
        print()

        print("Average:")
        if not stats['combined']['valid_count']:
            print("  not available -- both stages are needed for an average")
            print()
            continue
        print(f"  Valid: {stats['combined']['valid_count']}")
        print(f"  Mean: {stats['combined']['mean']:.2f}/100")
        print(f"  Range: [{stats['combined']['min']:.2f}, {stats['combined']['max']:.2f}]")
        print()


def export_to_csv(all_stats, output_file):
    """Write per-model statistics to ``output_file`` in CSV format."""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Total Samples',
            'InstFollow Valid', 'InstFollow Mean', 'InstFollow Min', 'InstFollow Max',
            'Quality Valid', 'Quality Mean', 'Quality Min', 'Quality Max',
            'Ave. Valid', 'Ave. Mean', 'Ave. Min', 'Ave. Max',
        ])

        for stats in all_stats:
            row = [
                stats['model_name'],
                stats['total_samples'],
                stats['stage2']['valid_count'],
                f"{stats['stage2']['mean'] * 100:.2f}",
                f"{stats['stage2']['min'] * 100:.2f}",
                f"{stats['stage2']['max'] * 100:.2f}",
                stats['stage3']['valid_count'],
                f"{stats['stage3']['mean']:.2f}",
                f"{stats['stage3']['min']:.2f}",
                f"{stats['stage3']['max']:.2f}",
                stats['combined']['valid_count'],
                f"{stats['combined']['mean']:.2f}",
                f"{stats['combined']['min']:.2f}",
                f"{stats['combined']['max']:.2f}",
            ]
            writer.writerow(row)

    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate PodBench evaluation results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing evaluation result files")
    parser.add_argument("--sort_by", type=str, default="combined",
                       choices=['combined', 'stage2', 'stage3'],
                       help="Leaderboard sort metric")
    parser.add_argument("--detailed", action="store_true",
                       help="Also print per-model statistics with the rubric breakdown")
    parser.add_argument("--export_csv", type=str, default=None,
                       help="Write the statistics to this CSV file")

    args = parser.parse_args()

    # Relative paths resolve against this file so the script runs from any directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = (args.results_dir if os.path.isabs(args.results_dir)
                   else os.path.join(script_dir, args.results_dir))

    eval_files = sorted(glob.glob(os.path.join(results_dir, "*_eval.jsonl")))
    if not eval_files:
        print(f"No evaluation result files found in: {results_dir}")
        return

    print(f"Found {len(eval_files)} evaluation result files")

    all_stats = []
    for eval_file in eval_files:
        model_name = os.path.basename(eval_file).replace('_eval.jsonl', '')
        print(f"Analyzing: {model_name}")
        try:
            all_stats.append(analyze_single_model(eval_file, model_name))
        except (OSError, ValueError, KeyError) as e:
            print(f"  Skipped ({type(e).__name__}: {e})")

    print_leaderboard(all_stats, sort_by=args.sort_by)

    if args.detailed:
        print_detailed_stats(all_stats)

    if args.export_csv:
        csv_path = (args.export_csv if os.path.isabs(args.export_csv)
                    else os.path.join(results_dir, args.export_csv))
        export_to_csv(all_stats, csv_path)


if __name__ == "__main__":
    main()
