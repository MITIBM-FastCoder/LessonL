from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from vllm import LLM, SamplingParams
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

from strategies.utils import *
from agents.utils import *
from agents.prompts.reflexion_opt_prompts import *

# from debate.reflexion_opt_prompts import *
# from debate.utils import *
import json
import hydra
import jsonlines
from omegaconf import DictConfig, OmegaConf


# @hydra.main(version_base=None, config_path="config", config_name="debate_config")
# def main(cfg : DictConfig) -> None:
def run(cfg: DictConfig) -> None:
    benchmark = cfg.benchmark

    max_iter = cfg.max_iter
    driver = cfg.driver
    evaldriver = cfg.evaldriver

    mode = cfg.mode

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""

    # localhost = "queue-g6e12xlarge-dy-g6e12xlarge-1"
    localhost = cfg.localhost
    MODEL_PATH = "Qwen/Qwen2.5-Coder-14B-Instruct"
    llm = OpenAI(
            base_url=f"http://{localhost}:8003/v1",
            api_key="token-2",
        )

    pass_at_k = cfg.pass_at_k
    temperature = cfg.temperature

    savename = f"{benchmark}_reflexion_{mode}_{max_iter}_{pass_at_k}.jsonl"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}"
    print("Reflexion Evaluator Results save path: ", evaluator_save_path)

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    in_tokens = 0
    out_tokens = 0

    print("Reflexion on Code Opt: ")

    # for problem in driver:
    for i, problem in enumerate_driver_resume(driver, evaluator_save_path):
        problem : LLM4PP_Problem
        print(i, problem.problem_id)




        cur_pass = 0
        code_to_submit = ""
        prev_speedup = -1
        while cur_pass < pass_at_k:
            print("cur_pass: ", cur_pass)
            prompt = generate_code_opt_prompt_code(problem.source_code, additional_package=additional_package)

            in_tokens += len(length_tokenizer.tokenize(prompt))

            output = llm.chat.completions.create(
                        model=MODEL_PATH,
                        messages=[
                        {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        top_p=0.95,
                        frequency_penalty=0.0
                    )
            
            optimized_code = output.choices[0].message.content

            

            out_tokens += len(length_tokenizer.tokenize(optimized_code))

            optimized_code = clean_output(response=optimized_code)
            if optimized_code == "":
                print("No code block found.")

            # submission = LLM4PP_Submission(problem=problem,
            #                         submitted_code=optimized_code)
            # # try:
            # #     response = evaldriverlist[0].submit(submission)
            # # except Exception as e:
            # #     print(f"skipping problem due to exception: {e}")
            # #     print("--- ParEval driver stdout ---")
            # #     print(response.stdout)


            # log, tag, speedup = process_execution_feedback(response.stdout)
            # if tag != "CORRECT":
            #     speedup = 0
            # if speedup > prev_speedup:
            #     prev_speedup = speedup
            #     code_to_submit = optimized_code
            
            cur_iter = 0

            while cur_iter < max_iter:
                print("cur_iter: ", cur_iter) 
                submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)
                try:
                    response = driver.submit(submission)
                except Exception as e:
                    print(f"skipping problem due to exception: {e}")
                    print("--- ParEval driver stdout ---")
                    print(response.stdout)
                log, tag, speedup = process_execution_feedback(response.stdout)
                faster_or_slower = ""
                if tag == "NOT_COMPILABLE":
                    log_lines = log.splitlines()  # Split log into lines
                    log = "\n".join(log_lines[:7]) if len(log_lines) > 7 else "\n".join(log_lines)
                
                if tag == "CORRECT":
                    if speedup >= 1.0 and speedup < 1.1:
                        faster_or_slower = "slightly faster"
                    elif speedup >= 1.1:
                        faster_or_slower = "significantly faster"
                else:
                    faster_or_slower = "slower"

                reflection_prompt = generate_reflection_prompt(src_code=problem.source_code,
                                                            tgt_code=optimized_code,
                                                            faster_or_slower=faster_or_slower,
                                                            correct=tag,
                                                            execution_feedback=log,
                                                            speedup=speedup)
                
                print("generating reflection. ")
                if len(reflection_prompt) > 3500:
                    reflection_prompt = reflection_prompt[:3500]
                in_tokens += len(length_tokenizer.tokenize(reflection_prompt))
                reflection = llm.chat.completions.create(
                        model=MODEL_PATH,
                        messages=[
                        {"role": "user", "content": reflection_prompt}
                        ],
                        temperature=temperature,
                        top_p=0.95,
                        frequency_penalty=0.5
                    )
            
                reflection = reflection.choices[0].message.content

                out_tokens += len(length_tokenizer.tokenize(reflection))
                                                                                
                print(reflection)

                optimized_prompt = generate_code_opt_with_reflection_prompt(src_code=problem.source_code,
                                                                        prev_code=optimized_code,
                                                                        self_reflection=reflection,
                                                                        execution_feedback=log,
                                                                        language="c++",
                                                                        additional_package=additional_package)
                in_tokens += len(length_tokenizer.tokenize(optimized_prompt))

                print("prompt with reflection. ")
                # --- Cap the input length to avoid exceeding model context window limits, comment out if not needed ---
                if len(optimized_prompt) > 3500:
                    optimized_prompt = optimized_prompt[:3500]
                 # --- Cap the input length to avoid exceeding model context window limits, comment out if not needed ---

                output = llm.chat.completions.create(
                        model=MODEL_PATH,
                        messages=[
                        {"role": "user", "content": optimized_prompt}
                        ],
                        temperature=temperature,
                        top_p=0.95,
                        frequency_penalty=0.0
                    )
                optimized_code = output.choices[0].message.content
                out_tokens += len(length_tokenizer.tokenize(optimized_code))
                optimized_code = clean_output(response=optimized_code)

                submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)
                    
                if tag != "CORRECT":
                    speedup = 0
                if speedup > prev_speedup:
                    prev_speedup = speedup
                    code_to_submit = optimized_code

                cur_iter += 1
            cur_pass += 1
        
        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=code_to_submit)
        try:
            response = evaldriver.submit(submission)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)
        
        evaldriver.save_one_response_jsonl(evaluator_save_path, [response.model_dump()], append=True)

    evaldriver.evaluate()

    print("In tokens: ", in_tokens)
    print("Out tokens: ", out_tokens)

# if __name__ == "__main__":
#     main()

