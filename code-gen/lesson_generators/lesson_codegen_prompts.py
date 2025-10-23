import json
import re
from .utils import *

### ================================ CODE GENERATION PROMPTS =============================== ###


def generate_python_code(func : str, language : str ="python") -> str:
    prompt_template = (
                "{instruction}\n\n### Here is the function to implement:\n{input}\n\n"
            )

    example_function_signature_and_docstring = '''def sum (a: float, b: float) -> float :\n  """ Return the sum of two floats a and b """'''

    example_completion = '''```python\ndef sum(a: float, b: float) -> float :\n  """ Return the sum of two floats a and b """\n  return a + b```'''

    instruction = f"You are given a function signature in {language} together with a docstring that explains what the function does. Your task is to implement the function according to the docstring. You should restate the function signature and docstring. Include the generated code between ```{language} and ```. For example, given the function signature and docstring\n\n{example_function_signature_and_docstring}\n\nYou should respond with\n\n{example_completion}\n"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : func})
    return prompt

def generate_python_code_with_lessons(func : str, lessons: list, language : str ="python") -> str:

    prompt_template = (
                "{instruction}\n\n### Here is the function to implement:\n{input}\n\nWhile you implement the function, consider the following lessons.\n{lessons}\n\n"
            )
    lessons_prompt = ""

    # idx_to_improve = ""
    # idx_opt = ""
    # idx_degrade = ""
    for i in range(len(lessons)):
        idx = i + 1
        lesson = lessons[i]
        lesson_content = lesson["lesson"]
        current_lesson = f"Lesson {idx} reasons why the code does not pass all test cases. {lesson_content}"
        lessons_prompt += current_lesson
    
    example_function_signature_and_docstring = '''def sum (a: float, b: float) -> float :\n  """ Return the sum of two floats a and b """'''

    example_completion = '''```python\ndef sum(a: float, b: float) -> float :\n  """ Return the sum of two floats a and b """\n  return a + b```'''

    instruction = f"You are given a function signature in {language} together with a docstring that explains what the function does. Your task is to implement the function according to the docstring. You should restate the function signature and docstring. Some lessons are provided to help you implement the function. Include the generated code between ```{language} and ```. For example, given the function signature and docstring\n\n{example_function_signature_and_docstring}\n\nYou should respond with\n\n{example_completion}\n"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : func, "lessons": lessons_prompt})
    return prompt

### ================================ CODE GENERATION PROMPTS END ====================================== ###

### ================================ LESSON GENERATION PROMPTS ========================================== ###


# def generate_lesson_correct_code(src_code : str, tgt_code: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following code you generated passed all test cases. Explain the reasons that make the code correct. Be brief in the explanations. Use only one or two sentences."
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code})
#     return prompt

def generate_lesson_partial_correct_code(gen_code: str, total_test_cases: int, passed_test_cases: int, failed_test_cases: int) -> str:
    prompt_template = (
        "{instruction}\n\n### Completed code:\n{gen_code}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"The following completed code is incorrect; i.e., it does not exactly reflect the description in the docstring. The code passes only {passed_test_cases} test cases out of {total_test_cases}, leaving {failed_test_cases} failed. Explain why the code is incorrect (that is, why it fails some test cases). Be brief in the explanations. Use only one or two sentences."
    prompt = prompt_template.format_map({"instruction" : instruction, "gen_code" : gen_code})
    return prompt

# def generate_lesson_incorrect_code(src_code : str, tgt_code: str, feedback: str) -> str:
#     prompt_template = (
#         "{instruction}\n\n### Code A:\n{codeA}\n\n### Code B:\n{codeB}\n\n### Execution Feedback:\n{feedback}\n\n"
#         )
#     # Speedup means how much faster comparing to the reference code.

#     instruction = f"The following two codes are not syntactically or semantically equivalent. Explain the reasons that make code B non-compilable in comparison to code A. Utilize the execution feedback provided after Code B. Be brief in the explanations. Use only one or two sentences."
#     prompt = prompt_template.format_map({"instruction" : instruction, "codeA" : src_code, "codeB": tgt_code, "feedback": feedback})
#     return prompt

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

### =================================== IDENTIFY LESSON PROMPTS END ========================================= ###

### =================================== MODIFY LESSON PROMPTS =============================================== ###

# NOTE: modify the prompt
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

### ==================================== MODIFY LESSON PROMPTS END ============================================ ###

### ==================================== SUMMARY PROMPTS ====================================================== ###

def summary_thoughts(input: str) -> str:
    prompt_template = (
        "{instruction}\n\n### Input:\n{input}\n\n"
        )
    # Speedup means how much faster comparing to the reference code.

    instruction = f"Preserve core ideas and summarize the following content in one or two sentences. Please use natural language instead of codes for all the summary."
    input = remove_triple_backtick_code_blocks(input_text=input)
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : input})
    return prompt

### =================================== SUMMARY PROMPTS END =================================================== ###



#NOTE: Better formatting
def format_reasoning(summary: str, feedback: str, reason: str) -> str:

    #prompt = f"If you perform the following:\n{summary}\nThe results will be: {feedback}\nBecause {reason}"
    prompt = f"The following case consists of the changes, the results, and the relevant analysis.\n{summary}\nAfter the changes that you have made, here are the results of the generated code from stdout: {feedback}\n{reason}" 
    return prompt