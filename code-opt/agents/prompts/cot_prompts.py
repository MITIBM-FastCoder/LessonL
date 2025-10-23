import re


def generate_code_opt_cot_prompt_code(src_code : str, language : str ="c++", additional_package : str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n"
            )

    instruction = f"You will be given a function written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```. Let's think step by step."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code})
    return prompt

