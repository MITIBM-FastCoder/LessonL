from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from client.chatapi import ChatAPI, MessageHistory
from vllm import LLM, SamplingParams
from openai import OpenAI
import json
import hydra
import jsonlines
import os

from omegaconf import DictConfig, OmegaConf
from agents.utils import *
from strategies.utils import *
from agents.prompts.base_prompts import *
from agents.utils import *

from transformers import AutoTokenizer, AutoModel

def run(cfg: DictConfig) -> None:
    benchmark = cfg.benchmark
    mode = cfg.mode

    temperature = cfg.temperature
    # Use benchmark drivers passed from main.py
    driver = cfg.driver
    # evaldriver = cfg.evaldriver
    trial = cfg.trial

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""

    #model = "gpt-4o"
    #model = "o3"
    #model = "gpt-4o-mini"

    model = cfg.model

    if model == "o3":
        reason = True
    else:
        reason = False

    #save_destination = f"./logs/{model}-{benchmark}-{mode}-lessons.jsonl"
    savename = f"openai_{model}_{benchmark}_{model}_{mode}_trial_{trial}.jsonl"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}.jsonl"
    print(f"OpenAI {model} Evaluator Results save path: ", evaluator_save_path)

    chatAPI = ChatAPI(temperature=temperature)

    print("start to evaluate ", model, trial)

    for i, problem in enumerate_driver_resume(driver, evaluator_save_path):
        problem : LLM4PP_Problem

        print(problem.problem_id)

        messages = MessageHistory()
        # messages.add_message("system", optimizer_prompt)
        prompt = generate_code_opt_prompt_code(src_code=problem.source_code, additional_package=additional_package)

        messages.add_message("user", prompt)

        response = chatAPI.get_response(model, messages, json_format=False, reason=reason)
        optimized_code = clean_output(response)
        
        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)

        try:
            response = driver.submit(submission)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)

        driver.save_one_response_jsonl(evaluator_save_path, [response.model_dump()], append=True)
    driver.evaluate()
    print(chatAPI.get_cost())
    out_tokens, input_tokens, total_tokens = chatAPI.get_usage()
    print("Output tokens: ", out_tokens[model])
    print("Input tokens: ", input_tokens[model])
    print("total tokens: ", input_tokens[model] + out_tokens[model])