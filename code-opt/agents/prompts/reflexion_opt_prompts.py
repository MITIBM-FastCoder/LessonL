import re

serial_contraints = "Compilers and environments between two codes are identical. The factor of compilers should not be considered in reasons."


def generate_code_opt_prompt_code(src_code : str, language : str ="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n"
            )

    instruction = f"You will be given a piece of code written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code})
    return prompt



def generate_reflection_prompt(src_code : str, tgt_code: str, faster_or_slower: str, correct: str, execution_feedback: str, speedup: int, constraints : str=serial_contraints) -> str:
    prompt_template = (
        "{instruction}\n\n// Code A:\n{codeA}\n\n// Code B:\n{codeB}\n\n// Execution Feedback of Code B:\n{execution_feedback}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"The following are the execution feedback and two codes. Code A is the source code yet to be optimized, and Code B is the code you generated to optimize Code A. They are compiled by using the same compiler and executed in the same environment. Code B is {correct} with a speedup {speedup}x. {constraints} Explain the reasons that make Code B is {correct} and run {faster_or_slower}.  Be brief in the explanations. Use only one or two sentences. You will need this as a hint when you try again later. Do not provide any code snippets. Execution feedback of Code B is attached.\n "
    prompt = prompt_template.format_map({"instruction" : instruction, "execution_feedback": execution_feedback, "codeA" : src_code, "codeB": tgt_code})
    return prompt

def generate_code_opt_with_reflection_prompt(src_code : str, prev_code: str, self_reflection: str, execution_feedback: str, language : str ="c++", constraints : str=serial_contraints, additional_package: str="") -> str:
    prompt_template = (
        "{instruction}\n\n// Previous Implmentation:\n{prev_code}\n\n// Executation Feedback of Previous Implementation:\n{execution_feedback}\n\n// Self Reflection:\n{self_reflection}\n\n// Code:\n{src_code}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"You will be given a function written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). You are given your previous implementation, the corresponding execution feedback, and your self-reflection. {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "prev_code": prev_code, "execution_feedback": execution_feedback, "self_reflection": self_reflection, "src_code": src_code})
    return prompt