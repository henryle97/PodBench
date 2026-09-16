#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate PodBench model outputs with an LLM judge served through OpenRouter.

Two rubrics are applied per sample: ``stage2`` scores instruction following against
a dynamically generated checklist (the paper's Stage 1), and ``stage3`` scores
podcast script quality on a 100-point rubric (the paper's Stage 2). Results are
appended as they complete, so an interrupted run resumes from its checkpoint.
"""

import json
import os
import argparse
import datetime
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from tqdm import tqdm

from podbench_rubric import DIMENSIONS, dimension_scores, find_dimension, read_score

try:
    import json_repair
except ImportError:
    json_repair = None
    print("[WARNING] 'json_repair' is not installed; malformed judge responses are "
          "more likely to be dropped, which biases scores. pip install json_repair")

file_lock = Lock()

# Judge used for all results reported in the paper.
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.5"

# Judge inputs are capped to bound cost and stay within context limits. The cap
# is in characters, so it buys a different amount of podcast in each language: a
# Chinese character is about one unit of speech, an English one about a fifth of
# a word. 16000 Chinese characters is 53 minutes at the rubric's own 300 per
# minute; 16000 English characters is about 2900 words, or 19 minutes at 150 per
# minute. Sized per language, the two are the same stretch of speech, and the
# English figure is then raised well past that so the cap stops binding on any
# realistic episode: 100000 characters is about two hours of speech.
#
# The Chinese figure stays at upstream's 16000. Raising it would quietly change
# the numbers this benchmark published.
#
# This matters more than a cost knob. Truncation keeps the head and drops the
# tail, so a script cut here loses its ending, and the rubric scores an ending
# that is not there: on a ten-episode English set, criterion 9 read 22% on the
# truncated scripts and 88% on the intact ones, while no other criterion moved
# more than 8 points.
MAX_SCRIPT_CHARS = 16000
MAX_SCRIPT_CHARS_BY_LANGUAGE = {"zh": 16000, "en": 100000}

# Quality rubric dimensions. The judge names them in the language of the prompt
# it was given, so the keys live in podbench_rubric rather than here.
# English names: Content Substance (45), Narrative Engagement (30),
# Conversational Naturalness (25).

# Judge prompts, per script language. --language picks the pair.
PROMPTS_BY_LANGUAGE = {
    "zh": ("evaluator/prompt_instruction_following.md",
           "evaluator/prompt_script_quality.md"),
    "en": ("evaluator/prompt_instruction_following.en.md",
           "evaluator/prompt_script_quality.en.md"),
}


#: Chat-completions endpoints a judge can be served from. OpenRouter is what the
#: paper used; any OpenAI-compatible endpoint works, which is how a judge that
#: OpenRouter does not carry is reached.
JUDGE_ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}

#: Endpoints that are not HTTP at all. `claude-cli` shells out to the local
#: `claude` binary in print mode and needs no key of its own.
CLI_ENDPOINT = "claude-cli"


class OpenRouterApi:
    """Minimal chat-completions client used for LLM-as-Judge scoring.

    Named for the endpoint the paper used. It speaks plain OpenAI-compatible
    chat completions, so `base_url` can point anywhere that shape is served.
    """

    def __init__(self, model, api_key=None, temperature=0.0, seed=42,
                 base_url=None, api_key_env="OPENROUTER_API_KEY"):
        self.base_url = base_url or JUDGE_ENDPOINTS["openrouter"][0]
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"A judge API key is required. "
                f"Set it via --api_key or the {api_key_env} environment variable."
            )
        self.model = model
        self.timeout = 600
        # Greedy decoding with a fixed seed keeps judge scores stable across
        # reruns. Both are omitted when None: some models reject temperature
        # outright (gpt-5.x returns 400 on any value but the default), and a
        # judge pinned that way is noisier than the paper's by construction.
        self.temperature = temperature
        self.seed = seed

    def call_chat(self, prompt, system_prompt=""):
        """Send one chat completion request and return the raw response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed

        return requests.post(url=self.base_url, headers=headers, json=payload,
                             timeout=self.timeout)


#: What the CLI judge is told it is. `claude -p` otherwise arrives wearing Claude
#: Code's agent system prompt, which is written for a coding session and names
#: tools the judge must not reach for. Replacing it outright is what makes the
#: CLI behave like the plain model the other endpoints reach.
CLI_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Follow the instructions in the user message "
    "exactly. Reply with only what they ask for, and nothing else -- no preamble, "
    "no commentary, no code fences unless the instructions call for them."
)


class _CliResponse:
    """A subprocess result, shaped like the chat-completions response.

    `extract_content_from_response` reads `status_code`, `text` and `json()`, so
    wearing that shape lets the CLI judge share every retry and parse path with
    the HTTP ones rather than growing a second one.

    Attributes:
        status_code: 200 when the CLI returned a completion, 500 otherwise.
        text: The error detail, read when `status_code` is not 200.
    """

    def __init__(self, status_code, text="", content=""):
        self.status_code = status_code
        self.text = text
        self._content = content

    def json(self):
        """The completion, in chat-completions shape."""
        return {"choices": [{"message": {"content": self._content}}]}


class ClaudeCliApi:
    """Judge served by the local `claude` CLI in print mode.

    Reaches a model no HTTP endpoint here carries, using whatever credentials
    the CLI already has. The prompt goes in on stdin rather than argv, because a
    rubric plus a 100000-character script is larger than a command line should
    carry.

    `temperature` and `seed` are accepted and ignored: the CLI exposes neither,
    so a judge run this way samples at the model's default and is noisier than a
    greedy one. Anything comparing its scores has to say so.
    """

    def __init__(self, model, temperature=None, seed=None, timeout=900,
                 system_prompt=CLI_JUDGE_SYSTEM_PROMPT):
        self.model = model
        self.timeout = timeout
        self.temperature = None
        self.seed = None
        self.system_prompt = system_prompt
        self.base_url = "claude -p (local CLI)"
        # A directory with nothing in it. The CLI discovers CLAUDE.md from its
        # working directory, and a judge that has read the repository's project
        # rules is not judging the artifact alone.
        self._cwd = tempfile.mkdtemp(prefix="podbench-judge-")

    def call_chat(self, prompt, system_prompt=""):
        """Run one judging prompt through the CLI.

        Args:
            prompt: The rendered rubric and artifact.
            system_prompt: Overrides the judge system prompt for this call.

        Returns:
            A `_CliResponse`. Failures come back as status 500 rather than
            raising, so the retry loop treats them like an HTTP error.
        """
        command = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",
            "--allowed-tools", "",
            "--system-prompt", system_prompt or self.system_prompt,
        ]
        try:
            completed = subprocess.run(
                command, input=prompt, capture_output=True, text=True,
                timeout=self.timeout, cwd=self._cwd, check=False,
            )
        except subprocess.TimeoutExpired:
            return _CliResponse(500, f"claude -p timed out after {self.timeout}s")
        except FileNotFoundError:
            return _CliResponse(500, "claude CLI not found on PATH")

        if completed.returncode != 0:
            return _CliResponse(500, f"claude -p exit {completed.returncode}: "
                                     f"{(completed.stderr or '')[:200]}")
        try:
            envelope = json.loads(completed.stdout)
        except ValueError:
            return _CliResponse(500, f"claude -p gave no JSON: {completed.stdout[:200]}")

        if envelope.get("is_error"):
            return _CliResponse(500, f"claude -p reported an error: "
                                     f"{str(envelope.get('result'))[:200]}")
        return _CliResponse(200, content=envelope.get("result") or "")


def load_prompt_template(prompt_file):
    """Read a judge prompt template."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def extract_answer_from_thinking(response):
    """Strip reasoning traces from a thinking model's output.

    Prefers the content of ``<answer>...</answer>``, falling back to whatever
    follows ``</think>``. Returns the input unchanged when neither tag is present.
    """
    answer_start = response.find('<answer>')
    answer_end = response.find('</answer>')
    if answer_start != -1 and answer_end != -1:
        return response[answer_start + len('<answer>'):answer_end].strip()

    think_end = response.find('</think>')
    if think_end != -1:
        answer = response[think_end + len('</think>'):].strip()
        if answer.startswith('<answer>'):
            answer = answer[len('<answer>'):].strip()
        if answer.endswith('</answer>'):
            answer = answer[:-len('</answer>')].strip()
        return answer

    return response


def _extract_raw_output(data):
    """Pull the generated text out of a record, tolerating the field names used by
    vLLM inference (``model_output``), direct API dumps (``model_response``), and
    raw provider payloads (``result_dict``)."""
    if 'result_dict' in data:
        choices = data['result_dict'].get('choices', [])
        if choices and 'message' in choices[0]:
            return choices[0]['message'].get('content', '')
    return data.get('model_response') or data.get('model_output') or ''


def load_model_results(input_file, is_thinking_model=False):
    """Load model outputs from a JSON array or a JSONL file.

    Samples without any generated text are skipped. For thinking models the
    reasoning trace is removed so only the final script is judged.
    """
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

    results = []
    for idx, data in enumerate(records):
        sample_id = data.get('id', str(idx))
        raw_output = _extract_raw_output(data)
        if not raw_output:
            print(f"Warning: no output for sample {sample_id}, skipping")
            continue

        results.append({
            'index': idx,
            'id': sample_id,
            'instruction': data.get('instruction', ''),
            'model_output': (extract_answer_from_thinking(raw_output)
                             if is_thinking_model else raw_output),
            'raw_data': data,
        })

    return results


def parse_json_robust(text):
    """Parse a judge response into JSON, tolerating common formatting noise.

    Tries the bare text, then ``json_repair`` if installed, then the outermost
    brace-balanced substring. Returns None if every strategy fails.
    """
    text = text.strip()

    if text.startswith('```json'):
        text = text[len('```json'):]
    elif text.startswith('```'):
        text = text[len('```'):]
    if text.endswith('```'):
        text = text[:-len('```')]
    text = text.strip()

    try:
        return json.loads(text)
    except ValueError:
        pass

    if json_repair:
        try:
            return json_repair.loads(text)
        except ValueError:
            pass

    start_idx = text.find('{')
    if start_idx != -1:
        depth = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            candidate = text[start_idx:end_idx + 1]
            try:
                return json.loads(candidate)
            except ValueError:
                if json_repair:
                    try:
                        return json_repair.loads(candidate)
                    except ValueError:
                        pass

    return None


def truncate_text(text, max_chars=MAX_SCRIPT_CHARS):
    """Truncate ``text`` to ``max_chars``, backing off to a nearby sentence end."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_boundary = max(truncated.rfind(mark)
                        for mark in ('。', '.', '！', '!', '？', '?', '\n\n'))
    # Only honour the boundary if it does not discard a large tail.
    if last_boundary > max_chars * 0.8:
        truncated = truncated[:last_boundary + 1]

    return truncated + "\n\n[Response truncated due to length...]"


def extract_content_from_response(rsp):
    """Return ``(content, error)`` from an OpenRouter response; error is None on success."""
    if rsp.status_code != 200:
        return None, f"HTTP {rsp.status_code}: {(rsp.text or '')[:200]}"

    result = rsp.json()
    choices = result.get('choices', [])
    if choices and 'message' in choices[0]:
        content = choices[0]['message'].get('content', '')
        if content:
            return content, None

    return None, f"Invalid response format: {str(result)[:200]}"


def _judge_with_retries(api, prompt, stage, validate, max_retries):
    """Query the judge until ``validate`` accepts the parsed response.

    Transport errors back off linearly, since they are usually rate limits;
    unparseable responses are retried at a fixed interval. Returns
    ``(parsed_result, None)`` on success or ``(None, last_error)`` on exhaustion.
    """
    label = f"[{stage}]"
    last_error = None

    for attempt in range(max_retries):
        print(f"{label} Attempt {attempt + 1}/{max_retries}...")
        try:
            content, error = extract_content_from_response(api.call_chat(prompt))

            if error:
                last_error = error
                print(f"{label} Error: {error}")
                if attempt < max_retries - 1:
                    backoff = 60 * (attempt + 1)
                    print(f"{label} Retrying in {backoff}s...")
                    time.sleep(backoff)
                continue

            parsed = parse_json_robust(content)
            if parsed is not None and validate(parsed):
                print(f"{label} Success")
                return parsed, None

            snippet = content[:200] if content else 'empty'
            last_error = f"Invalid response format: {snippet}"
            print(f"{label} Unparseable response: {snippet}")

        except requests.RequestException as e:
            last_error = f"RequestException: {e}"
            print(f"{label} Request failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(60)

    print(f"{label} All attempts failed")
    return None, last_error


def _annotate_truncation(eval_result, original_len, truncated_script):
    """Record that the judged script was shortened before scoring."""
    if len(truncated_script) < original_len:
        eval_result['truncated'] = True
        eval_result['original_length'] = original_len


def evaluate_stage2(api, instruction, podcast_script, prompt_template, max_retries=10,
                    max_script_chars=MAX_SCRIPT_CHARS):
    """Score instruction following against the judge-generated checklist.

    Returns ``(result, success)``. The result carries the raw checklist plus
    ``total_score``, ``max_score`` and ``normalized_score``; a failed evaluation is
    marked with negative scores so downstream analysis can exclude it.
    """
    if not podcast_script:
        return {
            "checklist": [],
            "total_score": -1,
            "max_score": 0,
            "normalized_score": -1,
            "error": "Empty podcast_script input",
        }, False

    truncated_script = truncate_text(podcast_script, max_script_chars)
    # Literal replacement avoids clashing with the JSON braces in the templates.
    prompt = (prompt_template
              .replace('{queries}', instruction or '')
              .replace('{podcast_script}', truncated_script))

    eval_result, last_error = _judge_with_retries(
        api, prompt, 'Stage2', lambda parsed: 'checklist' in parsed, max_retries)

    if eval_result is None:
        return {
            "checklist": [],
            "total_score": -1,
            "max_score": 0,
            "normalized_score": -1,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        }, False

    checklist = eval_result['checklist']
    total_score = sum(item.get('score', 0) for item in checklist)
    max_score = len(checklist)

    _annotate_truncation(eval_result, len(podcast_script), truncated_script)
    eval_result['total_score'] = total_score
    eval_result['max_score'] = max_score
    eval_result['normalized_score'] = total_score / max_score if max_score > 0 else 0

    return eval_result, True


def evaluate_stage3(api, podcast_script, prompt_template, max_retries=10,
                    max_script_chars=MAX_SCRIPT_CHARS):
    """Score podcast script quality on the 100-point rubric.

    Returns ``(result, success)``. ``total_score`` is the sum of the three
    dimension scores, or -1 when the evaluation could not be completed.
    """
    if not podcast_script:
        return {"total_score": -1, "error": "Empty podcast_script input"}, False

    truncated_script = truncate_text(podcast_script, max_script_chars)
    prompt = prompt_template.replace('{podcast_script}', truncated_script)

    # A judge that answered in a shape nobody asked for is retried, then failed.
    # Scoring it zero would be worse than losing it: a missing dimension used to
    # be skipped silently, so an answer in the wrong language scored 0/100 and
    # was recorded as a success.
    eval_result, last_error = _judge_with_retries(
        api, prompt, 'Stage3', lambda parsed: dimension_scores(parsed)[1], max_retries)

    if eval_result is None:
        return {
            "total_score": -1,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        }, False

    pairs, complete = dimension_scores(eval_result)
    if not complete:
        return {
            "total_score": -1,
            "error": ("Judge response is missing a rubric dimension. Read with "
                      "--language: a Chinese prompt names them in Chinese and an "
                      "English prompt in English."),
        }, False

    total_score = sum(score for score, _ in pairs)

    _annotate_truncation(eval_result, len(podcast_script), truncated_script)
    eval_result['total_score'] = total_score

    return eval_result, True


def evaluate_single_task(task_info):
    """Run one (sample, stage) evaluation for the thread pool."""
    stage = task_info['stage']
    if stage == 'stage2':
        eval_result, success = evaluate_stage2(
            task_info['api'], task_info['instruction'],
            task_info['model_output'], task_info['prompt_template'],
            max_script_chars=task_info['max_script_chars'])
    else:
        eval_result, success = evaluate_stage3(
            task_info['api'], task_info['model_output'], task_info['prompt_template'],
            max_script_chars=task_info['max_script_chars'])

    return task_info['task_key'], eval_result, success, task_info


def load_checkpoint(output_file):
    """Map sample index to its already-completed stage results, if any."""
    processed_items = {}
    if not os.path.exists(output_file):
        return processed_items

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                index = data['index']
            except (ValueError, KeyError):
                continue

            stage = data.get('stage')
            if stage:
                entry = processed_items.setdefault(index, {'stage2': None, 'stage3': None})
                entry[stage] = data.get('eval_result')
            else:
                processed_items[index] = {
                    'stage2': data.get('stage2_result'),
                    'stage3': data.get('stage3_result'),
                }
    return processed_items


def save_single_eval_result(output_file, index, sample_id, instruction, stage, eval_result, success,
                           target_model=None, judge_model=None):
    """Append one evaluation result under a lock so workers can write concurrently."""
    result_data = {
        'index': index,
        'id': sample_id,
        'instruction': instruction,
        'stage': stage,
        'eval_result': eval_result,
        'success': success,
        'timestamp': datetime.datetime.now().isoformat(),
        'target_model': target_model,
        'judge_model': judge_model
    }
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            f.flush()


def process_model_concurrent(api, model_results, output_file, stage2_template, stage3_template,
                            max_script_chars=MAX_SCRIPT_CHARS,
                            max_workers=20, resume=True, stages=('stage2', 'stage3'),
                            target_model=None, judge_model=None, limit=None):
    """Evaluate every (sample, stage) pair, writing results as they complete."""
    # Without resume the run starts from nothing, and that has to include the
    # file. Results are appended, so leaving an old file in place silently mixes
    # runs: a `--stages stage3` run over a file that already held stage2 rows
    # reports an instruction-following score nobody asked it to measure.
    if not resume and os.path.exists(output_file):
        backup = f"{output_file}.replaced"
        os.replace(output_file, backup)
        print(f"--no_resume: moved the previous results to {backup}")

    processed_items = load_checkpoint(output_file) if resume else {}

    if processed_items:
        completed_count = sum(1 for v in processed_items.values()
                            if (v.get('stage2') is not None or v.get('stage3') is not None))
        print(f"Resume from checkpoint: {completed_count} evaluations already completed")

    if limit is not None and limit > 0:
        model_results = model_results[:limit]
        print(f"Limiting processing to first {limit} samples")

    total_samples = len(model_results)

    print(f"\n{'='*60}")
    print("Evaluation Plan:")
    print(f"  Target Model: {target_model or 'Unknown'}")
    print(f"  Judge Model: {judge_model or 'Unknown'}")
    print(f"  Samples: {total_samples}")
    print(f"  Stages: {list(stages)}")
    print(f"  Script cap in force: {max_script_chars} characters")
    print(f"  Total API calls: {total_samples * len(stages)}")
    print(f"  Max concurrent workers: {max_workers}")
    print(f"{'='*60}\n")

    all_tasks = []
    for item in model_results:
        index = item['index']

        for stage in stages:
            if processed_items.get(index, {}).get(stage) is not None:
                continue

            task_info = {
                'api': api,
                'stage': stage,
                'instruction': item['instruction'],
                'model_output': item['model_output'],
                'prompt_template': stage2_template if stage == 'stage2' else stage3_template,
                'max_script_chars': max_script_chars,
                'task_key': (index, stage),
                'index': index,
                'sample_id': item['id'],
            }

            all_tasks.append(task_info)

    if not all_tasks:
        print("All evaluations already completed!")
        return

    print(f"Total tasks to process: {len(all_tasks)}")
    print(f"Starting concurrent evaluation with {max_workers} workers...\n")
    print(f"Results will be saved incrementally to: {output_file}\n")

    failed_tasks = []

    def run_evaluation_round(tasks, round_num=1):
        """Evaluate ``tasks`` in parallel and return counts plus the failures."""
        completed_count = 0
        failed_count = 0
        round_failed_tasks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_single_task, task): task for task in tasks}

            desc = f"Evaluating (Round {round_num})" if round_num > 1 else "Evaluating"
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        _, eval_result, success, task_info = future.result()

                        save_single_eval_result(
                            output_file,
                            index=task_info['index'],
                            sample_id=task_info['sample_id'],
                            instruction=task_info['instruction'],
                            stage=task_info['stage'],
                            eval_result=eval_result,
                            success=success,
                            target_model=target_model,
                            judge_model=judge_model
                        )

                        if success:
                            completed_count += 1
                        else:
                            failed_count += 1
                            round_failed_tasks.append(task)

                        pbar.update(1)
                        pbar.set_postfix({
                            'success': completed_count,
                            'failed': failed_count,
                            'success_rate': f'{completed_count/(completed_count+failed_count)*100:.1f}%'
                        })

                    except Exception as e:  # noqa: BLE001 - keep the pool alive
                        failed_count += 1
                        round_failed_tasks.append(task)
                        pbar.update(1)
                        print(f"\nTask failed with exception: {e}")

        return completed_count, failed_count, round_failed_tasks

    total_success, total_failed, failed_tasks = run_evaluation_round(all_tasks, round_num=1)

    # Whole rounds are retried after the per-call retries inside the judge helper,
    # which recovers from sustained rate limiting.
    max_retry_rounds = 3
    retry_round = 1
    while failed_tasks and retry_round <= max_retry_rounds:
        print(f"\n{'='*60}")
        print(f"Retry round {retry_round}: {len(failed_tasks)} failed tasks to retry...")
        print(f"{'='*60}\n")

        time.sleep(5)

        round_success, _, failed_tasks = run_evaluation_round(failed_tasks, round_num=retry_round+1)
        total_success += round_success
        total_failed = len(failed_tasks)
        retry_round += 1

    print(f"\n{'='*60}")
    print("Evaluation completed")
    print(f"  Total successful: {total_success}")
    print(f"  Final failed: {total_failed}")
    if total_success + total_failed > 0:
        print(f"  Final success rate: {total_success/(total_success+total_failed)*100:.2f}%")
    print(f"{'='*60}\n")

    print("Organizing results...")
    organize_results_file(output_file)


def organize_results_file(output_file):
    """Rewrite the result file with one record per (index, stage), sorted.

    Retried tasks append duplicate lines during a run; keeping the successful
    record per key prevents them from being counted twice during analysis.
    """
    if not os.path.exists(output_file):
        return

    results_by_key = {}

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                index = data['index']
                stage = data.get('stage')

                if stage:
                    key = (index, stage)
                    previous = results_by_key.get(key)
                    if previous is None or data.get('success') or not previous.get('success'):
                        results_by_key[key] = data
            except (ValueError, KeyError):
                continue

    sorted_keys = sorted(results_by_key.keys(), key=lambda x: (x[0], x[1]))

    # Write to a sibling file first so an interrupted rewrite cannot truncate results.
    temp_file = output_file + ".organized"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for key in sorted_keys:
            f.write(json.dumps(results_by_key[key], ensure_ascii=False) + '\n')

    os.replace(temp_file, output_file)
    print(f"Results organized: {len(sorted_keys)} evaluations saved to {output_file}")


def analyze_results(output_file):
    """Print summary statistics for a finished evaluation run."""
    print(f"\n{'='*60}")
    print("Analyzing evaluation results...")
    print(f"{'='*60}\n")

    results_by_index = {}

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                index = data['index']
            except (ValueError, KeyError):
                continue

            entry = results_by_index.setdefault(index, {'stage2': None, 'stage3': None})
            stage = data.get('stage')
            if stage:
                entry[stage] = data.get('eval_result')
            else:
                entry['stage2'] = data.get('stage2_result')
                entry['stage3'] = data.get('stage3_result')

    stage2_scores = []
    stage3_scores = []
    stage2_failed = 0
    stage3_failed = 0

    for data in results_by_index.values():
        stage2 = data.get('stage2')
        if stage2:
            if stage2.get('normalized_score', -1) >= 0:
                stage2_scores.append(stage2['normalized_score'])
            else:
                stage2_failed += 1

        stage3 = data.get('stage3')
        if stage3:
            if stage3.get('total_score', -1) >= 0:
                stage3_scores.append(stage3['total_score'])
            else:
                stage3_failed += 1

    print(f"Total samples: {len(results_by_index)}")
    print()

    print("Instruction Following:")
    print(f"  Valid evaluations: {len(stage2_scores)}")
    print(f"  Failed evaluations: {stage2_failed}")
    if stage2_scores:
        print(f"  Mean: {sum(stage2_scores)/len(stage2_scores)*100:.2f}/100")
        print(f"  Min: {min(stage2_scores)*100:.2f}")
        print(f"  Max: {max(stage2_scores)*100:.2f}")
    print()

    print("Podcast Script Quality:")
    print(f"  Valid evaluations: {len(stage3_scores)}")
    print(f"  Failed evaluations: {stage3_failed}")
    if stage3_scores:
        print(f"  Mean: {sum(stage3_scores)/len(stage3_scores):.2f}/100")
        print(f"  Min: {min(stage3_scores):.2f}")
        print(f"  Max: {max(stage3_scores):.2f}")
    print()

    if stage2_scores and stage3_scores:
        combined_scores = []
        for data in results_by_index.values():
            stage2 = data.get('stage2') or {}
            stage3 = data.get('stage3') or {}

            s2_score = stage2.get('normalized_score', -1)
            s3_score = stage3.get('total_score', -1)

            if s2_score >= 0 and s3_score >= 0:
                combined_scores.append(s2_score * 50 + s3_score * 0.5)

        if combined_scores:
            print("Average:")
            print(f"  Valid scores: {len(combined_scores)}")
            print(f"  Mean: {sum(combined_scores)/len(combined_scores):.2f}/100")
            print(f"  Min: {min(combined_scores):.2f}")
            print(f"  Max: {max(combined_scores):.2f}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PodBench model outputs with an LLM judge")
    parser.add_argument("--model_file", type=str, required=True,
                       help="Path to model results file (JSONL or JSON)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output evaluation results file")
    parser.add_argument("--is_thinking_model", action="store_true",
                       help="Strip reasoning traces before judging")
    parser.add_argument("--evaluator_model", type=str, default=DEFAULT_JUDGE_MODEL,
                       help=f"OpenRouter judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenRouter API key (defaults to $OPENROUTER_API_KEY)")
    parser.add_argument("--max_workers", type=int, default=20,
                       help="Number of concurrent judge requests")
    parser.add_argument("--limit", type=int, default=None,
                       help="Evaluate only the first N samples, for a cheap smoke test")
    parser.add_argument("--judge_temperature", type=float, default=0.0,
                       help="Judge decoding temperature; pass a negative value to "
                            "omit the field, which some models require")
    parser.add_argument("--judge_seed", type=int, default=42,
                       help="Judge sampling seed; pass -1 to omit the field")
    parser.add_argument("--endpoint", type=str, default="openrouter",
                       choices=sorted([*JUDGE_ENDPOINTS, CLI_ENDPOINT]),
                       help="Which endpoint serves the judge. claude-cli shells out "
                            "to the local `claude -p` and uses its credentials")
    parser.add_argument("--judge_timeout", type=int, default=900,
                       help="Seconds one judge call may take (claude-cli only)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="Override the endpoint URL entirely")
    parser.add_argument("--max_script_chars", type=int, default=None,
                       help="Cap on judged script length (default: by --language). "
                            "Truncation drops the tail, so a cap below the script's "
                            "length removes its ending before the rubric scores it")
    parser.add_argument("--language", type=str, default="zh",
                       choices=sorted(PROMPTS_BY_LANGUAGE),
                       help="Language of the scripts being judged; selects the judge "
                            "prompts. --stage2_prompt/--stage3_prompt override it")
    parser.add_argument("--no_resume", action="store_true",
                       help="Ignore any existing checkpoint and start fresh")
    parser.add_argument("--no_analysis", action="store_true",
                       help="Skip the summary printed after evaluation")
    parser.add_argument("--stages", type=str, default="stage2,stage3",
                       help="Comma-separated stages to run")
    parser.add_argument("--stage2_prompt", type=str, default=None,
                       help="Instruction-following prompt template (default: by --language)")
    parser.add_argument("--stage3_prompt", type=str, default=None,
                       help="Script-quality prompt template (default: by --language)")

    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(',')]

    default_stage2, default_stage3 = PROMPTS_BY_LANGUAGE[args.language]
    args.stage2_prompt = args.stage2_prompt or default_stage2
    args.stage3_prompt = args.stage3_prompt or default_stage3
    max_script_chars = (args.max_script_chars
                        or MAX_SCRIPT_CHARS_BY_LANGUAGE.get(args.language, MAX_SCRIPT_CHARS))

    # Relative paths resolve against this file so the script runs from any directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stage2_prompt_path = (args.stage2_prompt if os.path.isabs(args.stage2_prompt)
                          else os.path.join(script_dir, args.stage2_prompt))
    stage3_prompt_path = (args.stage3_prompt if os.path.isabs(args.stage3_prompt)
                          else os.path.join(script_dir, args.stage3_prompt))

    stage2_template = load_prompt_template(stage2_prompt_path)
    stage3_template = load_prompt_template(stage3_prompt_path)

    if args.endpoint == CLI_ENDPOINT:
        api = ClaudeCliApi(model=args.evaluator_model, timeout=args.judge_timeout)
    else:
        endpoint_url, api_key_env = JUDGE_ENDPOINTS[args.endpoint]
        api = OpenRouterApi(
            model=args.evaluator_model,
            api_key=args.api_key,
            temperature=None if args.judge_temperature < 0 else args.judge_temperature,
            seed=None if args.judge_seed < 0 else args.judge_seed,
            base_url=args.base_url or endpoint_url,
            api_key_env=api_key_env,
        )
    print(f"Judge: {args.evaluator_model} at {api.base_url}")
    print(f"  temperature: {'omitted' if api.temperature is None else api.temperature}, "
          f"seed: {'omitted' if api.seed is None else api.seed}")
    print(f"  prompts ({args.language}): {args.stage2_prompt}, {args.stage3_prompt}")

    model_results = load_model_results(args.model_file, args.is_thinking_model)
    print(f"Loaded {len(model_results)} samples from {args.model_file}")

    # Prefer the containing directory as the model label, since inference writes
    # results as <model>/<config>/generations.jsonl; fall back to the file name.
    target_model = os.path.basename(os.path.dirname(args.model_file))
    if not target_model or target_model == '.':
        target_model = os.path.basename(args.model_file).replace('.json', '').replace('.jsonl', '')

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    process_model_concurrent(
        api,
        model_results,
        args.output_file,
        stage2_template,
        stage3_template,
        max_script_chars=max_script_chars,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        stages=stages,
        target_model=target_model,
        judge_model=args.evaluator_model,
        limit=args.limit
    )
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    if not args.no_analysis and os.path.exists(args.output_file):
        analyze_results(args.output_file)


if __name__ == "__main__":
    main()
