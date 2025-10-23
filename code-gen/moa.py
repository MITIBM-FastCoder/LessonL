from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import random
from lesson_generators.agent_vllm import *
from lesson_generators.fastcoder.chatapi import ChatAPI, MessageHistory
from lesson_generators.lesson_codegen_prompts import generate_python_code
from typing import List, Dict, Type
from openai import OpenAI
from typing import List
import os
import math
import numpy as np
import torch
import logging
from logging import Logger
import gc


aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3


def get_final_system_prompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )

def run_llm(model, model_name, prompt, length_tokenizer, prev_response=None):
    """Run a single LLM call with a model while handling rate limits."""
    sleep_times = [1, 2, 4]  # Retry backoff times

    # for sleep_time in sleep_times:
    #     try:
    messages = [{"role": "user", "content": prompt}]
    final_system_prompt = ""
    if prev_response:
        final_system_prompt = get_final_system_prompt(aggregator_system_prompt, prev_response)
        if len(length_tokenizer.tokenize(prompt + final_system_prompt)) > 3500:
            final_system_prompt = final_system_prompt[:6000]
        messages.insert(0, {
            "role": "system",
            "content": final_system_prompt,
        })
    

    output = model.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            frequency_penalty=0,
        )
    in_tokens = len(length_tokenizer.tokenize(prompt + final_system_prompt))


    output = output.choices[0].message.content
    cur_func_impl = clean_output(output)
    out_tokens = len(length_tokenizer.tokenize(output))
    #optimized_code = clean_output(optimized_code)

    # response = requests.post(VLLM_API_URL, json=payload)
    # response_json = response.json()
    return cur_func_impl, in_tokens, out_tokens


def run_moa_gpt4o_aggregator(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    number_of_tests: int = 6
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    logger = logging.getLogger("main_logger")
    class_logger = logging.getLogger("class_logger")

    logger.setLevel(logging.INFO)
    class_logger.setLevel(logging.INFO)
    # model_list = []
    # for name in model_name:
    #     model_list.append(model_factory(name))
    # model = model_factory(model_name)

    #test_model = model_factory("gpt-4o")
    #test_model = model_factory(model_name[0])
    total_in_tokens = 0
    total_out_tokens = 0

    temperature = 0.2
    reason_temperature = 0.2
    # savename = f"{benchmark}_lesson_factor_{temperature}_{reason_temperature}_no_refutation.jsonl"
    # savedir = "logs"
    # os.makedirs(savedir, exist_ok=True)
    # save_destination = f"{savedir}/{savename}"
    # print("save name and destination: ", savename, save_destination)

    # Sampling Parameters
    # sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=2048)
    # reasoning_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024, repetition_penalty=1.5)

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    llm_1 = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-0",
    )


    llm_2 = OpenAI(
        base_url="http://localhost:8002/v1",
        api_key="token-1",
    )

    llm_3 = OpenAI(
        base_url="http://localhost:8003/v1",
        api_key="token-2",
    )

    reference_models = [llm_3, llm_2, llm_1]
    model_names = ["Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct", "deepseek-ai/deepseek-coder-7b-instruct-v1.5"]
    chatAPI = ChatAPI()
    print("Start MoA with GPT-4o:")

    # Load the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    # model = AutoModel.from_pretrained("neulab/codebert-cpp").to(device)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    #num_success = 112
    #print(num_success)
    # lesson_count = 0

    rounds = max_iters
    

    for i, item in enumerate_resume(dataset, log_path):
        #print(item["task_id"], i)
        #print(item.keys())
        # if item["name"] != "HumanEval_140_fix_spaces":
        #     continue
        # print("-----------Problem--------------")
        #print(item["name"], i)
        print(item["task_id"], i)
        # print("----------prompt----------------")
        # print(item["prompt"])
        # print("---------------SOLUTION--------")
        # print(item["canonical_solution"])
        # if item["name"] == "mbpp_260_newman_prime":
        #     continue
        
        round_completed = False
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = None
        # tests_i = item["test"]
        tests_i = []
        # tests_i = item["assertion"]
        #tests_i = item["test_list"]
        entry_point = item["entry_point"]
        
        #tests_i += gen.internal_tests(item["prompt"], test_model, 6)

        gen_prompt = generate_python_code(func=item["prompt"])

        # First layer - get initial responses one by one
        results = []

        for model, model_name in zip(reference_models, model_names):
            logger.info(f"Querying model: {model_name}")
            result, in_tokens, out_tokens = run_llm(model=model, model_name=model_name, prompt=gen_prompt, length_tokenizer=length_tokenizer)
            results.append(result)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens
            #lesson_counts = agent.context["lesson_counts"]
            cur_func_impl = result
            implementations.append(cur_func_impl)

        # Middle layers - refine responses sequentially
        for _ in range(1, layers - 1):
            new_results = []
            for model, model_name in zip(reference_models, model_names):
                logger.info(f"Refining response using model: {model_name}")
                result, in_tokens, out_tokens = run_llm(model=model, model_name=model_name, prompt=gen_prompt, length_tokenizer=length_tokenizer, prev_response=results)
                new_results.append(result)
                implementations.append(result)
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens
            results = new_results  # Update results for next iteration
            

        logger.info("Finalizing response with the aggregator model...")
        final_prompt = get_final_system_prompt(aggregator_system_prompt, results)
        
        messages = MessageHistory()
        # messages.add_message("system", optimizer_prompt)
        # messages.add_message("user", json.dumps({"solution.cpp": problem.source_code}))
        messages.add_message("system", final_prompt)
        messages.add_message("user", gen_prompt)

        response = chatAPI.get_response('gpt-4o', messages, json_format=False)
        cur_func_impl = clean_output(response)


        
        is_passing = exe.evaluate(
            item["entry_point"], cur_func_impl, item["test"], timeout=10)
        if is_passing:
            item["solution"] = cur_func_impl
            is_solved = True
            num_success += 1

        print("At round 4 num success, i: ", num_success, i, is_solved)
        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["acc"] = round(num_success/(i+1), 4)
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')