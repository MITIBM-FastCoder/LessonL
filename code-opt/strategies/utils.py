import jsonlines
import os
from typing import List
from client.models import LLM4PP_SubmissionResponse


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

def enumerate_driver_resume(driver, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(driver):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            # print(driver.responses)
            for item in reader:
                response = LLM4PP_SubmissionResponse(**item)
                driver.responses.append(response)
                count += 1
            # print(driver.responses)
            # print(count)

        for i, item in enumerate(driver):
            # print(item)
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item

def save_one_response_jsonl(path: str, responses: List[dict], append: bool = True):
    mode = "a" if append else "w"
    with jsonlines.open(path, mode) as writer:
        for response in responses:
            writer.write(response)