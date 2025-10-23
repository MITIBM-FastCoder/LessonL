from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from vllm import LLM, SamplingParams
from openai import OpenAI
import json
import hydra
import jsonlines
import os

from omegaconf import DictConfig, OmegaConf
from strategies.utils import *
from agents.utils import *
from agents.prompts.cot_prompts import *

from transformers import AutoTokenizer, AutoModel

def run(cfg: DictConfig) -> None:
    benchmark = cfg.benchmark
    mode = cfg.mode

    # if benchmark == "ParEval":
    #     driver = ParEvalDriver(mode)
    #     evaldriver = ParEvalDriver(mode)
    # elif benchmark == "PolyBench":
    #     driver = PolyBenchDriver(mode)
    #     evaldriver = PolyBenchDriver(mode)
    # else:
    #     print("Unknown Benchmark, program exits.")
    #     exit(0)

    temperature = cfg.temperature
    driver = cfg.driver

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""

    savename = f"cot-qwen14b_benchmark-{benchmark}_mode-{mode}"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}.jsonl"
    print("CoT Evaluator Results save path: ", evaluator_save_path)

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    in_tokens = 0
    out_tokens = 0

    localhost = cfg.localhost
    MODEL_PATH = "Qwen/Qwen2.5-Coder-14B-Instruct"
    llm = OpenAI(
            base_url=f"http://{localhost}:8003/v1",
            api_key="token-2",
        )

    #for problem in driver:
    for i, problem in enumerate_driver_resume(driver, evaluator_save_path):
        problem : LLM4PP_Problem
        print(problem.problem_id)

        # if problem.category == "search":
        #     continue
        
        prompt = generate_code_opt_cot_prompt_code(problem.source_code, additional_package=additional_package)
        in_tokens += len(length_tokenizer.tokenize(prompt))


        output = llm.chat.completions.create(
                    model=MODEL_PATH,
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0
                )
        
        optimized_code = output.choices[0].message.content

        out_tokens += len(length_tokenizer.tokenize(optimized_code))
        
        optimized_code = clean_output(response=optimized_code)
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
        #log, tag, speedup = pareval_process_execution_feedback(log=response.stdout)
        driver.save_one_response_jsonl(evaluator_save_path, [response.model_dump()], append=True)
    driver.evaluate()

    print("In tokens: ", in_tokens)
    print("Out tokens: ", out_tokens)