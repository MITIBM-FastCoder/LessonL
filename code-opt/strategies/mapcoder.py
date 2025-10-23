from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

from strategies.utils import *
from agents.utils import *
from agents.prompts.mapcoder_prompts import *


import json
import os
import hydra
import jsonlines
from omegaconf import DictConfig, OmegaConf

# def clean_output(response: str) -> str:
#     """
#     Extracts the code block from the given response text.
    
#     Args:
#         response (str): The generated response containing code wrapped in triple backticks.

#     Returns:
#         str: The extracted code block, or an empty string if no code block is found.
#     """
#     # Find the first and second occurrences of the code block delimiter
#     start = response.find("```cpp")
#     if start == -1:
#         return ""  # No code block found

#     # Find the next delimiter after the first one
#     end = response.find("```", start + 3)
#     if end == -1:
#         return ""  # No closing delimiter found

#     # Extract the code block
#     code_block = response[start + 6:end].strip()
#     return code_block

# @hydra.main(version_base=None, config_path="config", config_name="debate_config")
# def main(cfg : DictConfig) -> None:
def run(cfg: DictConfig) -> None:
    
    benchmark = cfg.benchmark

    #evaldriverlist = []
    # if benchmark == "ParEval":
    #     driver = ParEvalDriver()
    #     evaldriver = ParEvalDriver()
    #     for i in range(5+1):
    #         evaldriverlist.append(ParEvalDriver())
    # elif benchmark == "PolyBench":
    #     driver = PolyBenchDriver()
    #     evaldriver = PolyBenchDriver()
    #     for i in range(5+1):
    #         evaldriverlist.append(PolyBenchDriver())
    # else:
    #     print("Unknown benchmark. Program exits.")
    #     exit(0)
    driver = cfg.driver
    evaldriver = cfg.evaldriver

    mode = cfg.mode

    temperature = cfg.temperature
    k = cfg.k
    t = cfg.t

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
        additional_kb_prompt = f"You should include {mode}."
    else: #serial
        additional_package = ""
        additional_kb_prompt = ""

    savename = f"mapcoder_qwen14b_benchmark_{benchmark}_{mode}_{t}egs_{k}iter.jsonl"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}.jsonl"
    print("Mapcoder Evaluator Results save path: ", evaluator_save_path)

    language = "c++"

    localhost = cfg.localhost
    MODEL_PATH = "Qwen/Qwen2.5-Coder-14B-Instruct"
    llm = OpenAI(
            base_url=f"http://{localhost}:8003/v1",
            api_key="token-2",
        )


    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    in_tokens = 0
    out_tokens = 0

    
    # for problem in evaldriver:
    for _, problem in enumerate_driver_resume(evaldriver, evaluator_save_path):
        problem : LLM4PP_Problem
        print(problem.problem_id)

        # if problem.category == "search":
        #     continue
        # if problem.problem_id != "28_reduce_smallest_odd_number":
        #     continue


        source_code = problem.source_code

        while True:
            print("Generate KB")
            kb_prompt = generate_kb_prompt(src_code=source_code, k=k, additional_kb_prompt=additional_kb_prompt)

            in_tokens += len(length_tokenizer.tokenize(kb_prompt))

            output = llm.chat.completions.create(
                        model=MODEL_PATH,
                        messages=[
                        {"role": "user", "content": kb_prompt}
                        ],
                        temperature=temperature,
                        top_p=0.95,
                        frequency_penalty=0
                    )
            
            response = output.choices[0].message.content

            out_tokens += len(length_tokenizer.tokenize(response))


            
                # Post processing
            response = trim_text(
                response, "# Identify the optimization techniques that needs to be used to optimize the original code.")
            response = trim_text(
                response,"# Write a useful tutorial about the above optimization techniques. Provide a high level generic tutorial for optimizing this type of codes. Do not generate code.")
            response = trim_text(
                response, "# Planning to optimize this code: ")
            response = trim_text(
                response, f"# Let's think step by step to optimize this code in {language} programming language.")
            response = replace_tag(response, 'algorithm')
            response = replace_tag(response, 'description')
            response = replace_tag(response, 'code')
            response = replace_tag(response, 'planning')

            try:
                response = parse_xml(response)
                for example_no, example in enumerate(response["problem"], start=1):
                    example_problem = example["description"]
                    example_planning = example["planning"]
                break
            except:
                print("Failed to parse response into xml format. Continue generate response.")

        algorithm_prompt = f"## Relevant Optimization techniques to optimize the next code:\n{ response['algorithm']}"
        #sample_io_prompt = f"## Sample Test cases: \n{get_sample_io_str(item['sample_io'])}\n"
        plannings = []

        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]
            print("problem planning")
            input_for_problem_planning = generate_input_for_problem_planning(example_problem=example_problem, 
                                                                            example_planning=example_planning,
                                                                            algorithm_prompt=algorithm_prompt,
                                                                            src_code=source_code,
                                                                            additional_package=additional_package)
            in_tokens += len(length_tokenizer.tokenize(input_for_problem_planning))

            output = llm.chat.completions.create(
                    model=MODEL_PATH,
                    messages=[
                    {"role": "user", "content": input_for_problem_planning}
                    ],
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0
                )
            
            planning = output.choices[0].message.content

            out_tokens += len(length_tokenizer.tokenize(planning))

            print("Planning Verification")
            input_for_planning_verification = generate_input_for_planning_verification(src_code=source_code,
                                                                                    planning=planning)
            
            in_tokens += len(length_tokenizer.tokenize(input_for_planning_verification))
            
            output = llm.chat.completions.create(
                    model=MODEL_PATH,
                    messages=[
                    {"role": "user", "content": input_for_planning_verification}
                    ],
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0
                )
            
            verification_res = output.choices[0].message.content

            out_tokens += len(length_tokenizer.tokenize(verification_res))

            verification_res = replace_tag(verification_res, 'explanation')
            verification_res = replace_tag(verification_res, 'confidence')

            verification_res = parse_xml(verification_res)

            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())
            
            # print("verification res:")
            # print(verification_res['confidence'])

            plannings.append((
                planning,
                verification_res['confidence'],
                example
            ))
        
        plannings.sort(key=lambda x: x[1], reverse=True)

        code_to_submit = ""
        prev_speedup = -1

        result_speedup = [[0 for _ in range(6)] for _ in range(len(plannings))]
        result_code = [["" for _ in range(6)] for _ in range(len(plannings))]

        for idx_p, planning_with_ex in enumerate(plannings):
            planning, confidence, example = planning_with_ex

            print("Code Optimization")
            input_for_final_code_generation = generate_input_for_final_code_generation(src_code=source_code,
                                                                                    algorithm_prompt=algorithm_prompt,
                                                                                    planning=planning,
                                                                                    language=language,
                                                                                    additional_package=additional_package)
            in_tokens += len(length_tokenizer.tokenize(input_for_final_code_generation))

            # === Cap the input length to avoid exceeding model context window limits, Comment if not needed ===
            if len(input_for_final_code_generation) > 60000:
                input_for_final_code_generation = input_for_final_code_generation[:60000]
            # === Cap the input length to avoid exceeding model context window limits, Comment if not needed ===
            
            output = llm.chat.completions.create(
                    model=MODEL_PATH,
                    messages=[
                    {"role": "user", "content": input_for_final_code_generation}
                    ],
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0
                )
            
            optimized_code = output.choices[0].message.content
            out_tokens += len(length_tokenizer.tokenize(optimized_code))
            optimized_code = clean_output(optimized_code)

            submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)
            try:
                response = driver.submit(submission)
            except Exception as e:
                print(f"skipping problem due to exception: {e}")
                print("--- ParEval driver stdout ---")
                print(response.stdout)
            log, tag, speedup = process_execution_feedback(response.stdout)

            print("tag: ", tag)
            print("speedup: ", speedup)

            if tag != "CORRECT":
                speedup = 0

            if speedup > prev_speedup:
                prev_speedup = speedup
                code_to_submit = optimized_code
            
            result_speedup[idx_p][0] = speedup
            result_code[idx_p][0] = optimized_code

            for i in range(1, t+1):
                
                submission = LLM4PP_Submission(problem=problem,
                                        submitted_code=optimized_code)
                try:
                    response = driver.submit(submission)
                except Exception as e:
                    print(f"skipping problem due to exception: {e}")
                    print("--- ParEval driver stdout ---")
                    print(response.stdout)
                log, tag, speedup = process_execution_feedback(response.stdout)

                print("tag: ", tag)
                print("speedup: ", speedup)

                if tag != "CORRECT":
                    speedup = 0
                #     break

                print("Improving code optimization")
                if tag == "NOT_COMPILABLE":
                    log_lines = log.splitlines()  # Split log into lines
                    log = "\n".join(log_lines[:7]) if len(log_lines) > 7 else "\n".join(log_lines)
                
                if tag == "INCORRECT":
                    input_for_improving_code = generate_input_for_improving_correct_code(src_code=source_code,
                                                                                        algorithm_prompt=algorithm_prompt,
                                                                                        execution_feedback=log,
                                                                                        language=language)
                elif tag == "NOT_COMPILABLE":
                    input_for_improving_code = generate_input_for_improving_compilable_code(src_code=source_code,
                                                                                        algorithm_prompt=algorithm_prompt,
                                                                                        execution_feedback=log,
                                                                                        language=language)
                else:
                    input_for_improving_code = generate_input_for_improving_faster_code(src_code=source_code,
                                                                                        algorithm_prompt=algorithm_prompt,
                                                                                        execution_feedback=log,
                                                                                        language=language)
                    
                in_tokens += len(length_tokenizer.tokenize(input_for_improving_code))

                # === Cap the input length to avoid exceeding model context window limits, Comment if not needed ===
                if len(input_for_improving_code) > 60000:
                    input_for_improving_code = input_for_improving_code[:60000]
                # === Cap the input length to avoid exceeding model context window limits, Comment if not needed ===
                    
                output = llm.chat.completions.create(
                    model=MODEL_PATH,
                    messages=[
                    {"role": "user", "content": input_for_improving_code}
                    ],
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0
                )
            
                optimized_code = output.choices[0].message.content

                out_tokens += len(length_tokenizer.tokenize(optimized_code))
                optimized_code = clean_output(optimized_code)
                submission = LLM4PP_Submission(problem=problem,
                                        submitted_code=optimized_code)
                try:
                    response = driver.submit(submission)
                except Exception as e:
                    print(f"skipping problem due to exception: {e}")
                    print("--- ParEval driver stdout ---")
                    print(response.stdout)
                log, tag, speedup = process_execution_feedback(response.stdout)

                if tag != "CORRECT":
                    speedup = 0

                if speedup > prev_speedup:
                    prev_speedup = speedup
                    code_to_submit = optimized_code

                result_speedup[idx_p][i] = speedup
                result_code[idx_p][i] = optimized_code


        # Initialize a list to store the best code for each round
        # best_code_per_round = ["" for _ in range(6)]

        # For each round
        # for round_idx in range(6):
        #     print("round to submit: ", round_idx)
        #     max_speedup = 0
        #     best_plan_idx = 0
        #     code_to_submit_round = ""
            
        #     # Find the plan with the highest speedup for this round
        #     for plan_idx in range(len(planning)):
        #         try:
        #             if result_speedup[plan_idx][round_idx] > max_speedup:
        #                 max_speedup = result_speedup[plan_idx][round_idx]
        #                 best_plan_idx = plan_idx
        #                 code_to_submit_round = result_code[plan_idx][round_idx]
        #         except:
        #             continue

        #     submission = LLM4PP_Submission(problem=problem,
        #                                 submitted_code=code_to_submit_round)
            # try:
            #     response = evaldriverlist[round_idx].submit(submission)
            # except Exception as e:
            #     print(f"skipping problem due to exception: {e}")
            #     print("--- ParEval driver stdout ---")
            #     print(response.stdout)
            
            # Store the code from the best plan for this round
            # best_code_per_round[round_idx] = result_code[best_plan_idx][round_idx]
            
        


        #output = llm.generate(prompt, sampling_params)
        #optimized_code = output[0].outputs[0].text
        
        # optimized_code = clean_output(response=optimized_code)
        if optimized_code == "":
            print("No code block found.")
        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=code_to_submit)
        try:
            response = evaldriver.submit(submission)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)
        
        evaldriver.save_one_response_jsonl(evaluator_save_path, [response.model_dump()], append=True)


    #evaldriver.save_all_responses(f"./evaluator_results/{savename}.json")
    evaldriver.evaluate()

    # for r in range(len(evaldriverlist)):
    #     evdriver = evaldriverlist[r]
    #     print(f"Evaluate results from round {r}: ")
    #     evdriver.save_all_responses(f"./evaluator_results_rounds/{savename}_at_{r}_round.json")
    #     evdriver.evaluate()

    print("in tokens: ", in_tokens)
    print("out tokens: ", out_tokens)

# if __name__ == "__main__":
#     main()