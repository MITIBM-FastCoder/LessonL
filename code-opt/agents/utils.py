import re

def remove_triple_backtick_code_blocks(input_text):
    """
    Removes code blocks enclosed in triple backticks (```) from the input text.
    
    Args:
        input_text (str): The input string potentially containing code blocks.
    
    Returns:
        str: The input string with code blocks removed.
    """
    # Use a regular expression to remove content between triple backticks
    cleaned_text = re.sub(r'```.*?```', '', input_text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_first_explanation(input_text):
    """
    Extracts the first content between <explanation> and </explanation> tags.

    Args:
        input_text (str): The input string containing explanation tags.

    Returns:
        str: The first explanation content, or None if no explanation is found.
    """
    # Regular expression to match content between <explanation> and </explanation>
    match = re.search(r'<explanation>(.*?)</explanation>', input_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the first match with leading/trailing spaces removed
    return None

def contains_target_words(input_text):
    """
    Checks if the words 'But', 'but', 'However', or 'however' are present in the string.

    Args:
        input_text (str): The input string to search.

    Returns:
        bool: True if any of the words are found, False otherwise.
    """
    pattern = r'\b(But|but|However|however)\b'  # Matches exact words with word boundaries
    return bool(re.search(pattern, input_text))

def format_reasoning(summary: str, feedback: str, reason: str) -> str:

    #prompt = f"If you perform the following:\n{summary}\nThe results will be: {feedback}\nBecause {reason}"
    prompt = f"The following case consists of the changes, the results, and the relevant analysis.\n{summary}\nAfter the changes that you have made, here are the results of the generated code from stdout: {feedback}\n{reason}" 
    return prompt

def clean_output(response: str, language: str="c++") -> str:
    """ Extracts the code block from the given response text.
    
    Args:
        response (str): The generated response containing code wrapped in triple backticks.

    Returns:
        str: The extracted code block, or an empty string if no code block is found.
    """
    # Find the first and second occurrences of the code block delimiter
    start = response.find(f"```{language}")
    if start == -1:
        return ""  # No code block found

    # Find the next delimiter after the first one
    end = response.find("```", start + 3)
    if end == -1:
        return ""  # No closing delimiter found

    # Extract the code block
    code_block = response[start + 3 + len(language):end].strip()
    return code_block

def clean_output_analysis(output: str) -> str:
    """
    Cleans the given output by extracting the part starting from '### Target Code'.

    Args:
    output (str): The generated output containing code and other sections.

    Returns:
    str: The extracted portion starting from '### Target Code', or an empty string if not found.
    """
    target_keyword = "###"
    start = output.find(target_keyword)

    if start == -1:
    # If the keyword is not found, return an empty string
        return ""

    # Extract everything starting from '### Target Code'
    return output[:start].strip()


def extract_did_not_build_section(log: str):
    """
    Extracts the string between "----- DID NOT BUILD ----" and "--- CODE FILE ---" in the log.

    Args:
        log (str): The log text to parse.

    Returns:
        str: The extracted string, or an empty string if not found.
    """
    start_marker = "----- DID NOT BUILD ----"
    end_marker = "--- CODE FILE ---"
    
    # Find the start and end positions
    start = log.find(start_marker)
    if start == -1:
        return ""  # Start marker not found
    
    end = log.find(end_marker, start)
    if end == -1:
        return ""  # End marker not found
    
    # Extract the content between markers
    return log[start + len(start_marker):end].strip()


def process_execution_feedback(log: str):
    "Process the execution feedback from the ParEval Driver"
    if "DID NOT BUILD" in log:
        # The code did not compile
        log_info = extract_did_not_build_section(log)
        return log_info, "NOT_COMPILABLE", None
    elif "INCORRECT"in log:
        return log, "INCORRECT", None
    else:
        # Regular expression to find the speedup value
        speedup_match = re.search(r"speedup:\s*([\d.]+)", log)
        speedup = float(speedup_match.group(1))
        return log, "CORRECT", speedup