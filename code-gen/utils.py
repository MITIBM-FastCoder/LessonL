import os
import gzip
import json
import openai
import jsonlines

from typing import List
from evalplus.data import get_human_eval_plus, get_mbpp_plus

openai.api_key = os.getenv("OPENAI_API_KEY")


def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def write_jsonl(path: str, data: List[dict], append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


# generator that returns the item and the index in the dataset.
# if the results_path exists, it will skip all items that have been processed
# before.
def enumerate_resume(dataset, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item


def resume_success_count(dataset) -> int:
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count


def load_evalplus_dataset(benchmark: str) -> List[dict]:
    """
    Load dataset from evalplus package.

    Args:
        benchmark: Either 'humaneval' or 'mbpp'

    Returns:
        List of problem dictionaries compatible with existing format
    """
    if benchmark.lower() == 'humaneval':
        problems = get_human_eval_plus()
        is_mbpp = False
    elif benchmark.lower() == 'mbpp':
        problems = get_mbpp_plus()
        is_mbpp = True
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'humaneval' or 'mbpp'")

    # Convert evalplus format to existing format
    dataset = []
    for task_id, problem in problems.items():
        # HumanEval has 'test' field with check(candidate) function
        # MBPP has 'assertion' field with raw assert statements
        if is_mbpp:
            # MBPP: wrap assertions in check function
            assertions = problem.get('assertion', '')
            entry_point = problem['entry_point']
            test_code = f"def check(candidate):\n"
            for line in assertions.strip().split('\n'):
                if line.strip():
                    # Replace function name with candidate
                    line = line.replace(f"{entry_point}(", "candidate(")
                    test_code += f"    {line}\n"
            test_field = test_code
        else:
            # HumanEval: use test field directly
            test_field = problem.get('test', '')

        item = {
            'task_id': task_id,
            'prompt': problem['prompt'],
            'entry_point': problem['entry_point'],
            'canonical_solution': problem.get('canonical_solution', ''),
            'test': test_field,
        }

        dataset.append(item)

    return dataset

