import json
import re
from ..utils import *

constraints = "Compilers and environments between two codes are identical. The factor of compilers should not be considered in reasons."
### ================================ CODE OPTIMIZATION PROMPTS =============================== ###


def generate_code_opt_prompt_code(src_code : str, language : str ="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n"
            )

    instruction = f"You will be given a piece of code written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code})
    return prompt

def generate_code_opt_prompt_code_with_lessons(src_code : str, lessons: list, language : str ="c++", additional_package: str="") -> str:

    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\nWhile you rewrite the code, consider the following lessons. If Code A and Code B appear in the lessons, Code A refers to the given code and Code B refers to an attempted rewrite. Code B may not be optimal and it could be even worse than Code A.\n\n{lessons}\n\n"
            )

    lessons_prompt = ""
    for i in range(len(lessons)):
        idx = i + 1
        lesson = lessons[i]
        lesson_content = lesson["lesson"]
        if lesson["tag"] == "CORRECT":
            speedup = lesson["speedup"]
            if speedup >= 1 and speedup < 1.1:
                improves_or_degrades = "slightly improves"
                # idx_to_improve += f"{idx}, "
                current_lesson = f"\nLesson {idx} {improves_or_degrades} the code performance. {lesson_content}. However, despite the code performance improvement, the speedup is only marginal."
            elif speedup > 1.1:
                improves_or_degrades = "significantly improves"
                # idx_opt += f"{idx}, "
                current_lesson = f"\nLesson {idx} {improves_or_degrades} the code performance. {lesson_content}."
            else:
                improves_or_degrades = "degrades"
                # idx_degrade += f"{idx}, "
                current_lesson = f"\nLesson {idx} {improves_or_degrades} the code performance. {lesson_content}"
        elif lesson["tag"] == "INCORRECT":
            current_lesson = f"\nLesson {idx} compromises code equivalence. {lesson_content}"
        elif lesson["tag"] == "NOT_COMPILABLE":
            current_lesson = f"\nLesson {idx} produces non-compilable code. {lesson_content}"
        else:
            tag = lesson["tag"]
            print(f"Unrecognizable tag {tag}. Will keep prompting agent, but miss one lesson.")
        lessons_prompt += current_lesson
    
    lessons_prompt += f"\nBesides the above lessons, consider other optimization strategies that can more significantly improve the performance of the given code."

    
    instruction = f"You are given a piece of code written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the code. Some lessons regarding correctness and performance are provided to help you rewrite the code. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "lessons": lessons_prompt})
    return prompt

### ================================ CODE OPTIMIZATION PROMPTS END ====================================== ###

### ================================ LESSON GENERATION PROMPTS ========================================== ###
def generate_lesson_correct_tgt_code_prompt(src_code : str, tgt_code: str, faster_or_slower: str, speedup: int, constraints : str=constraints) -> str:
    prompt_template = (
        "{instruction}\n\n// Code A:\n{codeA}\n\n// Code B:\n{codeB}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"The following are two functionally equivalent codes. They are compiled by using the same compiler and executed in the same environment. Code B runs {faster_or_slower} than Code A with a speedup {speedup}x. Explain why Code B is {faster_or_slower}. Be brief in the explanations. Use only one or two sentences."
    
    prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code})
    
    return prompt


def generate_lesson_incorrect_tgt_code_prompt(src_code : str, tgt_code: str, constraints : str=constraints) -> str:
    prompt_template = (
        "{instruction}\n\n// Code A:\n{codeA}\n\n// Code B:\n{codeB}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"The following two codes are not functionally equivalent; that is, given the same input, they produce different outputs. Explain the reasons that make Code B nonequivalent to Code A. Be brief in the explanations. Use only one or two sentences."
    
    prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code})

    return prompt

def generate_lesson_not_compilable_tgt_code_prompt(src_code : str, tgt_code: str, feedback: str, constraints : str=constraints) -> str:
    prompt_template = (
        "{instruction}\n\n// Code A:\n{codeA}\n\n// Code B:\n{codeB}\n\nCompiler output:\n{feedback}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"The following are two codes. Code B attempts to improve the performance of Code A, but it has syntactic errors. Explain why Code B cannot be compiled. You may get hints from the compiler output provided after Code B. Be brief in the explanations. Use only one or two sentences."
    
    prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "feedback": feedback})
    
    return prompt

### ================================== LESSON GENERATION PROMPTS END ===================================== ###

### ================================== IDENTIFY LESSON PROMPTS =========================================== ###

# NOTE: modify the prompt
# def identify_lesson_correct_tgt_code(src_code : str, tgt_code: str, faster_or_slower: str, lesson: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation to evaluate:\n{lesson}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two functionally equivalent codes and an explanation of why code B runs {faster_or_slower} than code A. Evaluate the explanation and identify any errors or misconceptions. If you identify any such errors, please provide a short list of specific details and briefly discuss how the misconceptions can be fixed. If you do not identify any errors, say 'The explanation is correct.'"
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson})
#     return prompt

# # NOTE: modify the prompt
# def identify_lesson_incorrect_tgt_code(src_code : str, tgt_code: str, lesson: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation to evaluate:\n{lesson}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two functionally nonequivalent codes and an explanation of why code B is nonequivalent to code A. Evaluate the explanation and identify any errors or misconceptions. If you identify any such errors, please provide a short list of specific details and briefly discuss how the misconceptions can be fixed. If you do not identify any errors, say 'The explanation is correct.'"
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson})
#     return prompt

# # NOTE: modify the prompt
# def identify_lesson_non_compilable_tgt_code(src_code : str, tgt_code: str, lesson: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation to evaluate:\n{lesson}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two syntactically or semantically nonequivalent codes and an explanation of why code B is non-compilable in comparison to code A. Evaluate the explanation and identify any errors or misconceptions. If you identify any such errors, please provide a short list of specific details and briefly discuss how the misconceptions can be fixed. If you do not identify any errors, say 'The explanation is correct.'"
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson})
#     return prompt

# ### =================================== IDENTIFY LESSON PROMPTS END ========================================= ###

# ### =================================== MODIFY LESSON PROMPTS =============================================== ###

# # NOTE: modify the prompt
# def modify_lesson_correct_tgt_code(src_code : str, tgt_code: str, faster_or_slower: str, lesson: str, issues: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation:\n{lesson}\n\n### Issues with the explanation:\n{issues}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two functionally equivalent codes and an explanation of why code B runs {faster_or_slower} than code A. Given the list of possible issues, think about corrections to the explanation and directly modify the explanation. You should make as few changes as possible. Use only one or two sentences."
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson, "issues": issues})
#     return prompt

# # NOTE: modify the prompt
# def modify_lesson_incorrect_tgt_code(src_code : str, tgt_code: str, lesson: str, issues: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation:\n{lesson}\n\n### Issues with the explanation:\n{issues}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two functionally nonequivalent codes and an explanation of why code B is nonequivalent to code A. Given the list of possible issues, think about corrections to the explanation and directly modify the explanation. You should make as few changes as possible. Use only one or two sentences."
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson, "issues": issues})
#     return prompt

# # NOTE: modify the prompt
# def modify_lesson_non_compilable_tgt_code(src_code : str, tgt_code: str, lesson: str, issues: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Explanation:\n{lesson}\n\n### Issues with the explanation:\n{issues}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following are two syntactically or semantically nonequivalent codes and an explanation of why code B is non-compilable in comparison to code A. Given the list of possible issues, think about corrections to the explanation and directly modify the explanation. You should make as few changes as possible. Use only one or two sentences."
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "lesson": lesson, "issues": issues})
#     return prompt

# ### ==================================== MODIFY LESSON PROMPTS END ============================================ ###

# ### ==================================== SUMMARY PROMPTS ====================================================== ###

# def summary_thoughts(input: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Input:\n{input}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"Preserve core ideas and summarize the following content in one or two sentences. Please use natural language instead of codes for all the summary."
#     input = remove_triple_backtick_code_blocks(input_text=input)
#     prompt = prompt_template.format_map({"instruction" : instruction, "input" : input})
#     return prompt

# ### =================================== SUMMARY PROMPTS END =================================================== ###



# #NOTE: Better formatting
# def format_reasoning(summary: str, feedback: str, reason: str) -> str:

#     #prompt = f"If you perform the following:\n{summary}\nThe results will be: {feedback}\nBecause {reason}"
#     prompt = f"The following case consists of the changes, the results, and the relevant analysis.\n{summary}\nAfter the changes that you have made, here are the results of the generated code from stdout: {feedback}\n{reason}" 
#     return prompt