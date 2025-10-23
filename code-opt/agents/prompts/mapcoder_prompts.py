import re
import xml.etree.ElementTree as ET

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

def xml_to_dict(element):
    result = {}
    for child in element:
        if child:
            child_data = xml_to_dict(child)
            if child.tag in result:
                if isinstance(result[child.tag], list):
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = [result[child.tag], child_data]
            else:
                result[child.tag] = child_data
        else:
            result[child.tag] = child.text
    return result

def parse_xml(response: str) -> dict:
    if '```xml' in response:
        response = response.replace('```xml', '')
    if '```' in response:
        response = response.replace('```', '')

    try:
        root = ET.fromstring(response)
    except:
        try:
            root = ET.fromstring('<root>\n' + response + '\n</root>')
        except:
            root = ET.fromstring('<root>\n' + response)
    return xml_to_dict(root)

def trim_text(text: str, trimmed_text: str):
    return text.replace(trimmed_text, '').strip()

def replace_tag(text: str, tag: str):
    if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
        return text
    else:
        return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()




serial_contraints = "Compilers and environments between two codes are identical. The factor of compilers should not be considered in reasons."
### ================================ CODE OPTIMIZATION PROMPTS =============================== ###


def generate_code_opt_prompt_code(src_code : str, language : str ="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n"
            )

    instruction = f"You will be given a piece of code written in {language}. Your task is to rewrite it in the same language to improve its performance (i.e., execution time). {additional_package} Do not change the input/output behaviors of the function. Include the generated code between ```{language} and ```."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code})
    return prompt

def generate_kb_prompt(src_code: str, k: int, language: str ="c++", additional_kb_prompt: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Exemplars:\n{exemplars}\n\n// Algorithm:\n{algorithm}\n\n"
            )
    
    instruction = f"You will be given a function written in {language}. Provide relevant functions then identify the optimization techniques behind it and also explain the tutorial of the algorithm. {additional_kb_prompt}"

    exemplars = f"Recall {k} relevant and distinct problems (different from the problem mentioned above) about optimizing code. For each problem,\n1. describe it\n2. optimize {language} code step by step\n3. finally generate a planning to optimize the code.\n"

    algorithm = f"-----------------\nImportant:\nYour response must follow the following xml format.\n\n<root>\n<problem>\n# Recall {mapping[k]} relevant and distinct problems (different from problem mentioned above) about optimizing code. Write each problem in the following format.\n# <description>\n# Describe the problem.\n</description>\n<code>\n# Let's think step by step to optimize this code in {language} programming language. {additional_kb_prompt}\n</code>\n<planning>\n# Planning to optimize this code: \n</planning>\n</problem>\n\n# similarly add more problems here...\n\n<algorithm>\n# Identify the optimization techniques that needs to be used to optimize the original code.\n# Write a useful tutorial about the above optimization techniques. Provide a high level generic tutorial for optimizing this type of codes. Do not generate code.\n</algorithm>\n</root>\n"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "exemplars": exemplars, "algorithm": algorithm})
    return prompt

def generate_input_for_problem_planning(example_problem: str, example_planning: str, algorithm_prompt: str, src_code: str, language: str="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code: \n{example_problem}\n\n// Planning: \n{example_planning}\n{algorithm_prompt}\n\n// Code to be optimized:\n{input}\n\n// Planning:\n\n----------------\nImportant: You should give only the planning to optimize the code. Do not add extra explanation or words."
            )
    
    instruction = f"Given a function in {language} generate a concrete planning to optimize the code. {additional_package}"
    prompt = prompt_template.format_map({"instruction" : instruction, "example_problem": example_problem, "example_planning": example_planning, "algorithm_prompt": algorithm_prompt, "input" : src_code})
    return prompt

def generate_input_for_planning_verification(src_code: str, planning: str, language: str="c++") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Planning:\n{planning}\n\n----------------\nImportant: Your response must follow the following xml format-```\n<root>\n<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>\n<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>\n</root>\n```"
            )
    
    instruction = f"Given a function in {language} and a plan to optimize the code, tell whether the plan is correct and effective to optimize the code."
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "planning": planning})
    return prompt


def generate_input_for_final_code_generation(src_code: str, algorithm_prompt: str, planning: str, language: str="c++", additional_package: str="") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Planning:\n{planning}\n\nLet's think step by step.\n\n----------------\nImportant:\n## Your response must contain only the {language} code to solve this problem. Do not add extra explanation or words. Include the generated code between ```{language} and ```"
            )
    
    

    instruction = f"Given a function in {language} optimize the code. {algorithm_prompt} {additional_package}"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "planning": planning, "language": language})
    return prompt

def generate_input_for_improving_compilable_code(src_code: str, algorithm_prompt: str, execution_feedback: str, language: str="c++") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Execution Feedback:\n{execution_feedback}\n\n// Modified Planning:\nLet's think step by step.\n\n----------------\nImportant:\n## Your response must contain only the {language} code to optimize tghe cod. Your response must contain the modified planning and then the code. Include the generated code between ```{language} and ```"
            )
    
    # input_for_improving_code = [
    #     {
    #         "role": "user",
    #         "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
    #     }
    # ]

    instruction = f"Given a function in {language} optimize the code you have changed the code trying to make the code faster. But the generated code is not compilable.  Improve your code to solve the problem correctly. {algorithm_prompt}"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "execution_feedback": execution_feedback, "language": language})
    return prompt

def generate_input_for_improving_correct_code(src_code: str, algorithm_prompt: str, execution_feedback: str, language: str="c++") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Execution Feedback:\n{execution_feedback}\n\n// Modified Planning:\nLet's think step by step.\n\n----------------\nImportant:\n## Your response must contain only the {language} code to optimize tghe cod. Your response must contain the modified planning and then the code. Include the generated code between ```{language} and ```"
            )
    
    # input_for_improving_code = [
    #     {
    #         "role": "user",
    #         "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
    #     }
    # ]

    instruction = f"Given a function in {language} optimize the code you have changed the code trying to make the code faster. The generated code is compilable, but the code is not correct.  Improve your code to solve the problem correctly. {algorithm_prompt}"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "execution_feedback": execution_feedback, "language": language})
    return prompt

def generate_input_for_improving_faster_code(src_code: str, algorithm_prompt: str, execution_feedback: str, language: str="c++") -> str:
    prompt_template = (
                "{instruction}\n\n// Code:\n{input}\n\n// Execution Feedback:\n{execution_feedback}\n\n// Modified Planning:\nLet's think step by step.\n\n----------------\nImportant:\n## Your response must contain only the {language} code to optimize the code. Your response must contain the modified planning and then the code. Include the generated code between ```{language} and ```"
            )
    
    # input_for_improving_code = [
    #     {
    #         "role": "user",
    #         "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
    #     }
    # ]

    instruction = f"Given a function in {language} optimize the code you have changed the code trying to make the code faster. The generated code is correct, but the generated code is not fast enough.  Improve your code to solve the problem correctly. {algorithm_prompt}"
    prompt = prompt_template.format_map({"instruction" : instruction, "input" : src_code, "execution_feedback": execution_feedback, "language": language})
    return prompt