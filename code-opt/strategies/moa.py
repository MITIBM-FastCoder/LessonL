from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from client.chatapi import MessageHistory, ChatAPI

from typing import List, Dict, Type
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
import json
import torch
import numpy as np
import os
from scipy.stats import entropy
import time
from openai import OpenAI
import random
import math
import logging
from logging import Logger
import hydra
import jsonlines
import os

from omegaconf import DictConfig, OmegaConf

from agents.utils import *
from agents.agent_vllm import *
from strategies.utils import *

# import requests
import time
import logging

# aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

# Responses from models:"""


aggregator_system_prompt = """You have been provided with a set of optimized codes in c++ from various open-source models to the latest source code yet to be optimized. Your task is to synthesize these optimized codes into a single, high-quality code. It is crucial to critically evaluate the codes provided in these responses, recognizing that some of it may be not compilable, not correct, or not maintain good speedup. Your response should not simply replicate the given codes but should offer a refined, correct, and optimized code to the instruction. Ensure your code is well-structured, coherent, and adheres to the highest standards of correctness and fastest running time.

Responses from models:"""
layers = 3


def generate_code_opt_prompt_code(src_code : str, language : str ="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n"
            )

    instruction = f"You will be given a piece of code written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code})
    return prompt

def get_final_system_prompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


def run_llm(model, model_name, prompt, length_tokenizer, temperature, prev_response=None):
    """Run a single LLM call with a model while handling rate limits."""
    sleep_times = [1, 2, 4]  # Retry backoff times

    # for sleep_time in sleep_times:
    #     try:
    messages = [{"role": "user", "content": prompt}]
    final_system_prompt = ""
    if prev_response:
        final_system_prompt = get_final_system_prompt(aggregator_system_prompt, prev_response)
        # if len(length_tokenizer.tokenize(prompt + final_system_prompt)) > 3500:
        #     final_system_prompt = final_system_prompt[:6000]
        messages.insert(0, {
            "role": "system",
            "content": final_system_prompt,
        })
    try:
        output = model.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                frequency_penalty=0
            )
        in_tokens = len(length_tokenizer.tokenize(prompt + final_system_prompt))


        output = output.choices[0].message.content
        optimized_code = clean_output(output)
        out_tokens = len(length_tokenizer.tokenize(output))
    except:
        print("Synthesizing failured due to input length and limited context window of the agent. Directly optimize the code instead.")
        output = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=0.95,
                frequency_penalty=0
            )
        in_tokens = len(length_tokenizer.tokenize(prompt))


        output = output.choices[0].message.content
        optimized_code = clean_output(output)
        out_tokens = len(length_tokenizer.tokenize(output))

    return optimized_code, in_tokens, out_tokens
            
def run(cfg: DictConfig) -> None:
    """Run the MOA process sequentially."""
    

    benchmark = cfg.benchmark
    mode = cfg.mode
    T = cfg.Rounds # how many rounds for one specific problem
    layers = T

    driver = cfg.driver
    testdriver = cfg.evaldriver

    logging.config.dictConfig(cfg.logging)

    # A logger for this file
    logger = logging.getLogger("main_logger")
    class_logger = logging.getLogger("class_logger")

    temperature = cfg.temperature
    aggregator_temperature = cfg.aggregator_temperature
    mode = cfg.mode

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""

    localhost = cfg.localhost

    savename = f"{benchmark}_moa_gpt4o_aggregator_{mode}_{layers}.jsonl"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}"
    print("MoA Evaluator Results save path: ", evaluator_save_path)

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")


    # Initialize Agents
    llm_1 = OpenAI(
        base_url=f"http://{localhost}:8001/v1",
        api_key="token-0",
    )


    llm_2 = OpenAI(
        base_url=f"http://{localhost}:8002/v1",
        api_key="token-1",
    )

    llm_3 = OpenAI(
        base_url=f"http://{localhost}:8003/v1",
        api_key="token-2",
    )
    
    total_in_tokens = 0
    total_out_tokens = 0

    reference_models = [llm_1, llm_2, llm_3]
    model_names = ["deepseek-ai/deepseek-coder-7b-instruct-v1.5", "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct"]
    chatAPI = ChatAPI(temperature=aggregator_temperature)
    print("Start MoA with GPT-4o:")
    # for problem in driver:
    for i, problem in enumerate_driver_resume(driver, evaluator_save_path):
        problem : LLM4PP_Problem
        print(i, problem.problem_id)

        # First layer - get initial responses one by one
        results = []
        optimize_prompt = generate_code_opt_prompt_code(src_code=problem.source_code, additional_package=additional_package)
        for model, model_name in zip(reference_models, model_names):
            logger.info(f"Querying model: {model_name}")
            result, in_tokens, out_tokens = run_llm(model=model, model_name=model_name, prompt=optimize_prompt, length_tokenizer=length_tokenizer, temperature=temperature)
            results.append(result)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens

        code_to_submit = ""
        prev_speedup = -1
        for code in results:
            submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=code)
            try:
                response = testdriver.submit(submission)
            except Exception as e:
                print(f"skipping problem due to exception: {e}")
                print("--- ParEval driver stdout ---")
                print(response.stdout)
            log, tag, speedup = process_execution_feedback(log=response.stdout)
            if tag != "CORRECT":
                speedup = 0
            if speedup > prev_speedup:
                prev_speedup = speedup
                code_to_submit = code

        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=code_to_submit)


        # Middle layers - refine responses sequentially
        for idx in range(1, layers - 1):
            print("At Layer ", idx)
            new_results = []
            for model, model_name in zip(reference_models, model_names):
                logger.info(f"Refining response using model: {model_name}")
                result, in_tokens, out_tokens = run_llm(model=model, model_name=model_name, prompt=optimize_prompt, length_tokenizer=length_tokenizer, temperature=temperature, prev_response=results)
                new_results.append(result)
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens
            results = new_results  # Update results for next iteration


            code_to_submit = ""
            prev_speedup = -1
            for code in results:
                submission = LLM4PP_Submission(problem=problem,
                                        submitted_code=code)
                try:
                    response = testdriver.submit(submission)
                except Exception as e:
                    print(f"skipping problem due to exception: {e}")
                    print("--- ParEval driver stdout ---")
                    print(response.stdout)
                log, tag, speedup = process_execution_feedback(log=response.stdout)
                if tag != "CORRECT":
                    speedup = 0
                if speedup > prev_speedup:
                    prev_speedup = speedup
                    code_to_submit = code

            submission = LLM4PP_Submission(problem=problem,
                                        submitted_code=code_to_submit)

        # Final aggregation step
        logger.info("Finalizing response with the aggregator model...")
        final_prompt = get_final_system_prompt(aggregator_system_prompt, results)
        
        messages = MessageHistory()
        messages.add_message("system", final_prompt)
        messages.add_message("user", optimize_prompt)

        response = chatAPI.get_response('gpt-4o', messages, json_format=False)
        optimized_code = clean_output(response)

        # total_in_tokens += in_tokens
        # total_out_tokens += out_tokens

        if optimized_code == "":
            print("No code block found.")
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

    print("Total in tokens: ", total_in_tokens)
    print("Total out tokens: ", total_out_tokens)
    print("Total tokens open source model: ", total_in_tokens + total_out_tokens)
    total_tokens = total_in_tokens + total_out_tokens
    price =  (total_tokens / 3) * 2 * 0.2 / 1000000 + total_tokens / 3 * 0.3 / 1000000
    print("Open Source Model Price: ", price)
    print("=============== OPENAI MODEL ========================:")
    print(chatAPI.get_cost())
    out_tokens, input_tokens, total_tokens = chatAPI.get_usage()
    print("Output tokens: ", out_tokens['gpt-4o'])
    print("Input tokens: ", input_tokens['gpt-4o'])
    print("total tokens: ", input_tokens['gpt-4o'] + out_tokens['gpt-4o'])


# main
# if __name__ == "__main__":
#     main()
