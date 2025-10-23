from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.driver import LLM4PP_Driver
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from strategies.utils import enumerate_driver_resume, save_one_response_jsonl


from typing import List, Dict, Type, Optional, Callable
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
from omegaconf import DictConfig, OmegaConf



from agents.utils import *
from agents.agent_vllm import *

# Load the model and move it to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute magnitudes (L2 norms)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    return float(dot_product / (norm_vec1 * norm_vec2))


def get_embedding(tokenizer, model, input: str):
    # Tokenize the explanation
    inputs = tokenizer(input, return_tensors="pt", truncation=True, padding=True).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

    return cls_embedding

def select_high_speedup(all_lessons_codes: List[Dict[str, any]], count: int):
    """
    Given lessons with codes (Z_all), select k/2 of lessons out of the whole Z_all with good speedup (>= 1.1)

    Args:
        all_lessons_codes (List[Dict[str, any]]): Z_all, lessons with codes and speedup
        count (int): k/2, the number of lessons chosen with high speedup
    
    Returns:
        List[Dict[str, any]]: lessons with codes, speedup and other information, where all lessons has high speedup
    """

    lessons_codes_high_speedup = [lesson for lesson in all_lessons_codes if lesson["speedup"] * lesson["factor"] >= 1.1]
    lessons_codes_remain = [lesson for lesson in all_lessons_codes if lesson["speedup"] * lesson["factor"] < 1.1]

    if len(lessons_codes_high_speedup) > count:
        lessons_codes_high_speedup = sorted(lessons_codes_high_speedup, key=lambda x: x["speedup"] * x["factor"], reverse=True)
        lessons_codes_remain += lessons_codes_high_speedup[count:]
        lessons_codes_high_speedup = lessons_codes_high_speedup[:count]
    
    return lessons_codes_high_speedup, lessons_codes_remain

def select_high_quality(lessons_codes_remain: List[Dict[str, any]], count: int):
    """
    Given remaining low-speedup lessons with codes (Z_remain), select k/2 of lessons out of the whole Z_remain with good quality by cosine similarity

    Args:
        lessons_codes_remain (List[Dict[str, any]]): Z_remain, low speedup lessons with codes, assume quality is already available for each element of dictionary
        count (int): k/2, the number of lessons chosen with high quality
    
    Returns:
        List[Dict[str, any]]: lessons with codes, speedup and other information, where all lessons has high speedup
    """

    lessons_codes_high_quality = sorted(lessons_codes_remain, key=lambda x: x["quality"], reverse=True)[:count]
    lessons_codes_remain = sorted(lessons_codes_remain, key=lambda x: x["quality"], reverse=True)[count:]
    return lessons_codes_high_quality, lessons_codes_remain

def get_code_speedup_lesson(problem: LLM4PP_Problem, agent_list: List[VLLMAgent], length_tokenizer: AutoTokenizer, logger: Logger, driver: LLM4PP_Driver, t: int, lesson_count: int, all_lessons_codes: list):
    source_code = problem.source_code
    lessons_codes = [] # lessons and codes with embedding, the current Z
    optimized_code_list = []
    submission_list=[]
    response_list = []
    lesson_list = []
    source_code_list = []
    speedup_list = []
    lesson_counts = agent_list[0].context["lesson_counts"]

    total_in_tokens = 0
    total_out_tokens = 0
    
    # ================= Round 0:  GET Code, Speedup and Lessons ==================================
    logger.info("round: %d", t)
    # Get Code from Agents, corresponding submission and response of generated code, and Lessons of generated code.
    for i in range(len(agent_list)):
        agent = agent_list[i]
        optimized_code, in_tokens, out_tokens = agent.optimize_code(context=agent.context)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        optimized_code_list.append(optimized_code)
        submission = LLM4PP_Submission(problem=problem, submitted_code=optimized_code)
        submission_list.append(optimized_code)
        try:
            response = driver.submit(submission)
        except Exception as e:
            print(f"submission {i+1} error")
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)
        response_list.append(response)

        # Generate Lesson
        agent.context["tgt_code"] = response.submission.submitted_code
        agent.context["feedback"] = response.stdout
        lesson, in_tokens, out_tokens = agent.generate_lesson(context=agent.context)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        lesson_length = length_tokenizer.tokenize(lesson)
        logger.info("Lesson %d length : %d", i, len(lesson_length))

        lesson_list.append(lesson)
        logger.debug("Lesson %d content: %s\n", i, lesson)

        agent.context["lesson"] = lesson
        source_code_list.append(agent.context["src_code"])

        log, tag, speedup = process_execution_feedback(response.stdout)

        if tag != "CORRECT":
            speedup = 0
        speedup_list.append(speedup)

        lesson_count += 1
        lesson_info_dict = {
            "lesson": lesson,
            "tag": tag,
            "log": log,
            "speedup": speedup,
            "src_code": source_code,
            "tgt_code": optimized_code,
            "idx": i,
            "quality": -1,
            "round": t,
            "factor": 1,
            "lesson_count": lesson_count,
            "used_lessons": agent.context["lessons"]
        }
        lessons_codes.append(lesson_info_dict)
    
    # Factor Adjustment
    if t > 0:
        for lesson in all_lessons_codes:
            factor = 0
            lesson_count = lesson.get("lesson_count")
            if lesson_count not in lesson_counts:
                continue
            # if lesson_count in lesson_counts:
            prev_speedup = lesson.get("speedup")
            for curr_sp in speedup_list:
                if curr_sp > prev_speedup:
                    factor += 1.1
                else:
                    factor += 0.9
            factor = factor / len(speedup_list)
            lesson["factor"] = factor
    
    return lessons_codes, lesson_count, all_lessons_codes, total_in_tokens, total_out_tokens


# @hydra.main(version_base=None, config_path="config/strategy", config_name="lesson_config")
# def main(cfg : DictConfig) -> None:
def run(cfg: DictConfig) -> None:

    benchmark = cfg.benchmark
    T = cfg.Rounds # how many rounds for one specific problem

    logging.config.dictConfig(cfg.logging)

    # A logger for this file
    logger = logging.getLogger("main_logger")
    class_logger = logging.getLogger("class_logger")

    T = cfg.Rounds # how many rounds for one specific problem
    k = cfg.k # how many lessons to put in agents for learning

    temperature = cfg.temperature
    reason_temperature = cfg.reason_temperature
    mode = cfg.mode

    # Use benchmark drivers passed from main.py
    driver = cfg.driver
    evaldriver = cfg.evaldriver

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""


    localhost = cfg.localhost
    print(localhost, T, benchmark)
    savename = f"lesson_benchmark-{benchmark}_mode-{mode}_temperature-{temperature}_reason-temperature-{reason_temperature}_Rounds-{T}_k-{k}"
    savedir = "lessonlogs"
    os.makedirs(savedir, exist_ok=True)
    save_destination = f"./{savedir}/{savename}.jsonl"
    print("LessonL Logs save name and destination: ", savename, save_destination)

    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}.jsonl"
    print("LessonL Evaluator Results save path: ", evaluator_save_path)
    

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")


    # Initialize Agents
    llm_1 = OpenAI(
        base_url=f"http://{localhost}:8001/v1",
        api_key="token-0",
    )
    
    agent_1 = VLLMAgent(llm=llm_1, model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)


    llm_2 = OpenAI(
        base_url=f"http://{localhost}:8002/v1",
        api_key="token-1",
    )

    agent_2 = VLLMAgent(llm=llm_2, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)
    llm_3 = OpenAI(
        base_url=f"http://{localhost}:8003/v1",
        api_key="token-2",
    )

    agent_3 = VLLMAgent(llm=llm_3, model_name="Qwen/Qwen2.5-Coder-14B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)
    agent_list = [agent_1, agent_2, agent_3]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    model = AutoModel.from_pretrained("neulab/codebert-cpp").to(device)

    lessons_history = []

    logger.info("Start")

    count = 0
    lesson_count = 0
    total_in_tokens = 0
    total_out_tokens = 0

    #for problem in evaldriver:
    for i, problem in enumerate_driver_resume(evaldriver, evaluator_save_path):
        problem : LLM4PP_Problem
            
        #if problem.problem_id != "11_geometry_convex_hull_perimeter":
        #    continue
            
        # if problem.problem_id != "10_geometry_convex_hull":
        #     continue

        # if problem.problem_id != "28_reduce_smallest_odd_number":
        #     continue

        # if problem.problem_id != "06_fft_dft":
        #     continue

        # if problem.problem_id != "13_geometry_closest_pair_2d":
        #     continue

        # if problem.problem_id != "06_fft_dft" and problem.problem_id != "13_geometry_closest_pair_2d":
        #     continue

        logger.info(problem.problem_id)
        logger.info(problem.category)
       

        # if problem.category != 'scan':
        #     continue

        source_code = problem.source_code
        all_lessons_codes = [] # the Z_all
        rounds_speedup_list = []
        
        # For each problem, initialize the context that is used to prompt LLM
        for agent in agent_list:
            agent.context = {
                "src_code": source_code,
                "tgt_code": "",
                "feedback": "",
                "lesson": "", # the lesson corresponds to the generated code, i.e. tgt_code
                "lessons": "", # the lessons put in the prompt for future rounds
                "issues": "",
                "lesson_counts": [],
                "problem_id": problem.problem_id,
                "category": problem.category
            }
        
        code_to_submit = ""
        

        current_lessons_codes, lesson_count, all_lessons_codes, in_tokens, out_tokens = get_code_speedup_lesson(problem=problem, 
                                                                      agent_list=agent_list, 
                                                                      length_tokenizer=length_tokenizer, 
                                                                      logger=logger, 
                                                                      driver=driver, 
                                                                      t=0, 
                                                                      lesson_count=lesson_count,
                                                                      all_lessons_codes=all_lessons_codes)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        corrected_lessons_codes = current_lessons_codes

        source_code_list = [item["src_code"] for item in corrected_lessons_codes]
        corrected_lesson_list = [item["lesson"] for item in corrected_lessons_codes]

        source_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=source_code_list)
        corrected_lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=corrected_lesson_list)

        for i in range(len(corrected_lessons_codes)):
            quality = cosine_similarity(corrected_lesson_emb[i], source_code_emb[i])
            corrected_lessons_codes[i]["quality"] = float(quality)
        
        all_lessons_codes += corrected_lessons_codes

        sorted_lessons_codes = sorted(all_lessons_codes, key=lambda x: x["speedup"], reverse=True)

        # ================ Round 0 Finished ===============================
        for t in range(1, T):
            logger.info("round: %d", t)

            if t * len(agent_list) <= k:
                lessons_codes_next_round = all_lessons_codes
                unselected_lessons = []
            else:
                lessons_codes_high_speedup, lessons_codes_remain = select_high_speedup(all_lessons_codes, count=math.ceil(k/2))
                num_high_speedup = len(lessons_codes_high_speedup)
                num_high_quality = k - num_high_speedup
                lessons_codes_low_speedup, lessons_codes_remain = select_high_quality(lessons_codes_remain, count=num_high_quality)
                lessons_codes_next_round = lessons_codes_high_speedup + lessons_codes_low_speedup
            

            sorted_lessons_codes = sorted(lessons_codes_next_round, key=lambda x: x["speedup"], reverse=True)

            logger.info("Best Speedup After Round %d : %f", t, sorted_lessons_codes[0]["speedup"])
            rounds_speedup_list.append(sorted_lessons_codes[0]["speedup"])
            
            #new_source_code = sorted_lessons_codes[0]["src_code"]
            new_source_code = source_code
            source_code = new_source_code

            lesson_to_agent = [{"lesson": item["lesson"], "tag": item["tag"], "speedup": item["speedup"], "lesson_count": item["lesson_count"], "idx": item["idx"], "round": item["round"], "quality": item["quality"], "factor": item["factor"], "lesson_count": item["lesson_count"]} for item in lessons_codes_next_round]

            # Extract lesson_count corresponding to each lesson in lesson_to_agent
            lesson_counts = [
                item["lesson_count"] for item in lesson_to_agent
            ]
            # Update agent history and source code
            for agent in agent_list:
                agent.single_round_memory.append(agent.context)
                agent.context = {
                "src_code": new_source_code,
                "tgt_code": "",
                "feedback": "",
                "lesson": "",
                "lessons": lesson_to_agent,
                "issues": "",
                "lesson_counts": lesson_counts,
                "problem_id": problem.problem_id,
                "category": problem.category
            }
            
            current_lessons_codes, lesson_count, all_lessons_codes, in_tokens, out_tokens = get_code_speedup_lesson(problem=problem, 
                                                                          agent_list=agent_list, 
                                                                          length_tokenizer=length_tokenizer, 
                                                                          logger=logger, 
                                                                          driver=driver, 
                                                                          t=t,
                                                                          lesson_count=lesson_count,
                                                                          all_lessons_codes=all_lessons_codes)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens

            source_code_list = [item["src_code"] for item in current_lessons_codes]
            current_lesson_list = [item["lesson"] for item in current_lessons_codes]

            source_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=source_code_list)
            current_lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=current_lesson_list)

            for i in range(len(current_lessons_codes)):
                quality = cosine_similarity(current_lesson_emb[i], source_code_emb[i])
                current_lessons_codes[i]["quality"] = float(quality)
            
            all_lessons_codes += current_lessons_codes
            sorted_lessons_codes = sorted(all_lessons_codes, key=lambda x: x["speedup"], reverse=True)

        sorted_lessons_codes = sorted(all_lessons_codes, key=lambda x: x["speedup"], reverse=True)
        logger.info("Best Speedup After Whole Process: %f", sorted_lessons_codes[0]["speedup"])
        rounds_speedup_list.append(sorted_lessons_codes[0]["speedup"])

        code_to_submit = sorted_lessons_codes[0]["tgt_code"]
        submission_best = LLM4PP_Submission(problem=problem, submitted_code=code_to_submit)
        try:
            response_submit = evaldriver.submit(submission_best)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response_submit.stdout)
        
        single_lesson_dict = {
            "lessons": sorted_lessons_codes,
            "problem_id": problem.problem_id,
            "category": problem.category,
            "rounds_speedup_list": rounds_speedup_list
        }

        lessons_history.append(single_lesson_dict)

        save_one_response_jsonl(save_destination, [single_lesson_dict], append=True)

        evaldriver.save_one_response_jsonl(evaluator_save_path, [response_submit.model_dump()], append=True)

        count += 1


    # evaldriver.save_all_responses(f"./evaluator_results/{savename}_test.json")
    evaldriver.evaluate()

    print("Total in tokens: ", total_in_tokens)
    print("Total out tokens: ", total_out_tokens)

    # for r in range(len(evaldriverlist)):
    #     evdriver = evaldriverlist[r]
    #     print(f"Evaluate results from round {r}: ")
    #     evdriver.save_all_responses(f"./evaluator_results_rounds/{savename}_{r}.json")
    #     evdriver.evaluate()

    # for item in lessons_history:
    #     for item2 in item["lessons"]:
    #         if isinstance(item2.get("quality"),np.float32):
    #             item2["quality"] = float(item2["quality"])
    #         for item3 in item2["used_lessons"]:
    #             if isinstance(item3.get("quality"), np.float32):
    #                 item3["quality"] = float(item3["quality"])
            

    # with jsonlines.open(save_destination, mode="w") as writer:
    #     writer.write_all(lessons_history)  # Writes all dictionaries as separate lines
    # # for agent in agent_list:
    #     agent.save_memory()

# if __name__ == "__main__":
#     main()

# Function to get the Hydra config function that can be called from main.py
