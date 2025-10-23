from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from lesson_generators.agent_vllm import *
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute magnitudes (L2 norms)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Compute cosine similarity
    return float(dot_product / (norm_vec1 * norm_vec2))

def count_elements(state):
    total_count = len(state)
    false_count = state.count(False)
    passed_count = state.count(True)
    return total_count, passed_count, false_count

def get_embedding(tokenizer, model, input: str):
    # Tokenize the explanation
    inputs = tokenizer(input, return_tensors="pt", truncation=True, padding=True).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

    del outputs, inputs

    return cls_embedding

def select_high_passed_cases(all_lessons_codes: List[Dict[str, any]], count: int, total_cases: int):
    """
    Given lessons with codes, select 'count' lessons with highest scaled passed test cases.
    
    Args:
        all_lessons_codes (List[Dict[str, any]]): Lessons with codes, passed test cases and factor
        count (int): Number of lessons to select with highest scaled passed test cases
        total_cases (int): Total number of test cases
    
    Returns:
        Tuple[List[Dict[str, any]], List[Dict[str, any]]]: 
            - First list contains selected lessons with highest scaled passed test cases
            - Second list contains remaining lessons
    """
    # Sort all lessons by their scaled passed test cases (passed_test_cases * factor)
    sorted_lessons = sorted(all_lessons_codes, 
                          key=lambda x: x["passed_test_cases"] * x["factor"],
                          reverse=True)
    
    # Select top 'count' lessons
    selected_lessons = sorted_lessons[:count]
    
    # Get remaining lessons
    remaining_lessons = sorted_lessons[count:]
    
    return selected_lessons, remaining_lessons

def select_high_quality(lessons_codes_remain: List[Dict[str, any]], count: int):
    """
    Given remaining low-speedup lessons with codes (Z_remain), select k/2 of lessons out of the whole Z_remain with good quality by cosine similarity

    Args:
        lessons_codes_remain (List[Dict[str, any]]): Z_remain, low speedup lessons with codes, assume quality is already available for each element of dictionary
        count (int): k/2, the number of lessons chosen with high quality
    
    Returns:
        List[Dict[str, any]]: lessons with codes, speedup and other information, where all lessons has high speedup
    """

    #optimized_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=optimized_code_list)

    lessons_codes_high_quality = sorted(lessons_codes_remain, key=lambda x: x["quality"], reverse=True)[:count]
    return lessons_codes_high_quality


def run_multi_agent_lessons_opensource(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    number_of_tests: int = 2
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    logger = logging.getLogger("main_logger")
    class_logger = logging.getLogger("class_logger")

    logger.setLevel(logging.INFO)
    class_logger.setLevel(logging.INFO)
    model_list = []
    for name in model_name:
        model_list.append(model_factory(name))

    
    test_model = model_factory(model_name[0]) # Choose the biggest model to generate test cases

    temperature = 0.2
    reason_temperature = 0.2

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    llm_1 = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-0",
    )
    
    agent_1 = VLLMAgent(llm=llm_1, model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, logger=class_logger)


    llm_2 = OpenAI(
        base_url="http://localhost:8002/v1",
        api_key="token-1",
    )

    agent_2 = VLLMAgent(llm=llm_2, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, logger=class_logger)

    llm_3 = OpenAI(
        base_url="http://localhost:8003/v1",
        api_key="token-2",
    )

    agent_3 = VLLMAgent(llm=llm_3, model_name="Qwen/Qwen2.5-Coder-14B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, logger=class_logger)

    agent_list = [agent_3, agent_2, agent_1] # we order the agents by their model size, from large to small

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-python")
    model = AutoModel.from_pretrained("neulab/codebert-python").to(device)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    lesson_count = 0

    rounds = max_iters
    

    for i, item in enumerate_resume(dataset, log_path):

        print(f'Processing {i+1}/{num_items}: {item["task_id"]}')
        
        round_completed = False
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = None
        tests_i = []

        entry_point = item["entry_point"]

        all_lessons_codes = [] # the Z_all
        # For each problem, initialize the context that is used to prompt LLM
        for agent in agent_list:
            agent.context = {
                "func": item["prompt"],
                "code": "",
                "feedback": "",
                "lesson": "", # the lesson corresponds to the generated code, i.e. tgt_code
                "lessons": "", # the lessons put in the prompt for future rounds
                "issues": "",
                "total_test_cases_num": 0,
                "passed_test_cases_num": 0,
                "failed_test_cases_num": 0,
                "lesson_counts": [],
            }
        
        # ================================= ROUND 0 ==========================================
        lesson_list = []
        func_list = []
        current_lessons_codes = []
        pass_test_cases_list = []


        tests_i += gen.internal_tests(item["prompt"], test_model, 6)

        print_v("round: ", 0)

        for j in range(len(agent_list)):
            agent = agent_list[j]
            lesson_counts = agent.context["lesson_counts"]
            cur_func_impl, _, _ = agent.generate_code(context=agent.context)
            implementations.append(cur_func_impl)
            agent.context["code"] = cur_func_impl

            is_passing, feedback, state = exe.execute(cur_func_impl, tests_i)

            if is_passing:
                is_passing = exe.evaluate(
                    entry_point, cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                item["is_solved"] = is_solved
                #item["reflections"] = reflections
                item["lessons"] = all_lessons_codes # empty at round 0
                item["implementations"] = implementations
                item["test_feedback"] = test_feedback
                item["solution"] = cur_func_impl
                item["acc"] = round(num_success/(i+1), 4)
                write_jsonl(log_path, [item], append=True)
                round_completed = True

                print_v(
                    f'At round 0 completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')
                break

            total_test_cases, passed_count, failed_count = count_elements(state)
            print("test cases number: total, passed, failed ", total_test_cases, passed_count, failed_count)
            agent.context["total_test_cases"] = total_test_cases
            agent.context["passed_test_cases"] = passed_count
            agent.context['failed_test_cases'] = failed_count
            lesson, _, _ = agent.generate_lesson(context=agent.context)

            agent.context["lesson"] = lesson

            lesson_list.append(lesson)
            func_list.append(item["prompt"])
            pass_test_cases_list.append(passed_count)
            lesson_count += 1
            lesson_info_dict = {
                "lesson": lesson,
                "passed_test_cases": passed_count,
                "code": cur_func_impl,
                "func": item["prompt"],
                "idx": j,
                "quality": -1,
                "round": 0,
                "factor": 1,
                "lesson_count": lesson_count
            }
            current_lessons_codes.append(lesson_info_dict)

        if round_completed:
            print("Already completed")
            continue

        func_list = [item["func"] for item in current_lessons_codes]
        lesson_list = [item["lesson"] for item in current_lessons_codes]

        func_emb = get_embedding(tokenizer=tokenizer, model=model, input=func_list)
        lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=lesson_list)

        for j in range(len(current_lessons_codes)):
            quality = cosine_similarity(lesson_emb[j], func_emb[j])
            current_lessons_codes[j]["quality"] = quality

        all_lessons_codes += current_lessons_codes

        # ================================= ROUND 1 and more ==========================================
        for t in range(1, rounds):
            print_v("round: ", t)
            if t * len(agent_list) <= 4: #NOTE: we only have 3 agents for now
                lessons_codes_next_round = all_lessons_codes
            else:
                # Selection of lessons
                lessons_codes_high_speedup, lessons_codes_remain = select_high_passed_cases(all_lessons_codes, count=math.ceil(4/2), total_cases=total_test_cases)
                num_high_speedup = len(lessons_codes_high_speedup)
                num_high_quality = 4 - num_high_speedup
                lessons_codes_low_speedup = select_high_quality(lessons_codes_remain, count=num_high_quality)
                lessons_codes_next_round = lessons_codes_high_speedup + lessons_codes_low_speedup
            
            sorted_lessons_codes = sorted(lessons_codes_next_round, key=lambda x: x["passed_test_cases"], reverse=True)

            print_v("Best Passed test cases at Round %d : %f", t, sorted_lessons_codes[0]["passed_test_cases"])

            lesson_to_agent = [{"lesson": item["lesson"], "passed_test_cases": item["passed_test_cases"], "lesson_count": item["lesson_count"]} for item in lessons_codes_next_round]

            # Extract lesson_count corresponding to each lesson in lesson_to_agent
            lesson_counts = [
                item["lesson_count"] for item in lesson_to_agent
            ]
            # Update agent history and source code
            for agent in agent_list:
                #agent.single_round_memory.append(agent.context)
                agent.context = {
                "func": item["prompt"],
                "code": "",
                "feedback": "",
                "lesson": "", # the lesson corresponds to the generated code, i.e. tgt_code
                "lessons": lesson_to_agent, # the lessons put in the prompt for future rounds
                "issues": "",
                "total_test_cases_num": 0,
                "passed_test_cases_num": 0,
                "failed_test_cases_num": 0,
                "lesson_counts": lesson_counts,
            }
                
            
            lesson_list = []
            func_list = []
            current_lessons_codes = []
            pass_test_cases_list = []
            for j in range(len(agent_list)):
                agent = agent_list[j]
                lesson_counts = agent.context["lesson_counts"]
                cur_func_impl, _, _ = agent.generate_code(context=agent.context)
                implementations.append(cur_func_impl)
                agent.context["code"] = cur_func_impl
                is_passing, _, state = exe.execute(cur_func_impl, tests_i)
                # if solved, exit early
                if is_passing:
                    is_passing = exe.evaluate(
                        entry_point, cur_func_impl, item["test"], timeout=10)
                    is_solved = is_passing
                    num_success += int(is_passing)
                    item["is_solved"] = is_solved
                    item["lessons"] = all_lessons_codes # all lessons so far
                    item["implementations"] = implementations
                    item["test_feedback"] = test_feedback
                    item["solution"] = cur_func_impl
                    item["acc"] = round(num_success/(i+1), 4)
                    write_jsonl(log_path, [item], append=True)
                    round_completed = True

                    print_v(
                        f'At round {t} completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')
                    break

                total_test_cases, passed_count, failed_count = count_elements(state)
                print_v("test cases number: total, passed, failed ", total_test_cases, passed_count, failed_count)
                agent.context["total_test_cases"] = total_test_cases
                agent.context["passed_test_cases"] = passed_count
                agent.context['failed_test_cases'] = failed_count
                lesson, _, _ = agent.generate_lesson(context=agent.context)

                agent.context["lesson"] = lesson

                lesson_list.append(lesson)
                func_list.append(item["prompt"])
                pass_test_cases_list.append(passed_count)
                lesson_count += 1
                lesson_info_dict = {
                    "lesson": lesson,
                    "passed_test_cases": passed_count,
                    "code": cur_func_impl,
                    "func": item["prompt"],
                    "idx": j,
                    "quality": -1,
                    "round": t,
                    "factor": 1,
                    "lesson_count": lesson_count
                }
                current_lessons_codes.append(lesson_info_dict)

            if round_completed: # if solved, exit early
                # print_v("Already completed")
                break

            func_list = [item["func"] for item in current_lessons_codes]
            lesson_list = [item["lesson"] for item in current_lessons_codes]
            func_emb = get_embedding(tokenizer=tokenizer, model=model, input=func_list)
            lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=lesson_list)

            for j in range(len(current_lessons_codes)):
                quality = cosine_similarity(lesson_emb[j], func_emb[j])
                current_lessons_codes[j]["quality"] = quality
            all_lessons_codes += current_lessons_codes

            # Update factor for all lessons in Z_all
            for lesson in all_lessons_codes:
                factor = 0
                lesson_count = lesson.get("lesson_count")
                if lesson_count not in lesson_counts:
                    continue
                prev_passed = lesson.get("passed_test_cases")
                for curr_passed in pass_test_cases_list:
                    if curr_passed > prev_passed:
                        factor += 1.1
                    else:
                        factor += 0.9
                factor = factor / len(pass_test_cases_list)
                lesson["factor"] = factor
        if round_completed:
            #print("Already completed")
            continue

        sorted_lessons_codes = sorted(all_lessons_codes, key=lambda x: x["passed_test_cases"], reverse=True)
        code_to_submit = sorted_lessons_codes[0]["code"]
        is_passing = exe.evaluate(
            item["entry_point"], code_to_submit, item["test"], timeout=10)
        if is_passing:
            item["solution"] = code_to_submit
            is_solved = True
            num_success += 1

        # print("At round 4 num success, i: ", num_success, i, is_solved)
        
        item["is_solved"] = is_solved
        item["lessons"] = sorted_lessons_codes
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = code_to_submit
        item["acc"] = round(num_success/(i+1), 4)
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'At round 4 completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')
        
        # At the end of each main problem loop
        torch.cuda.empty_cache()
        gc.collect()

