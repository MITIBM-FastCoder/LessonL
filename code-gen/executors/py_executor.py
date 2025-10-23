import ast
import signal
import astunparse
import re

from .executor_utils import function_with_timeout

from typing import List
from .executor_types import ExecuteResult, Executor

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:

                function_with_timeout(exec, (func_test_list[i], globals()), timeout)

                success_tests += [tests[i]]
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # output: {output}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)

        feedback = "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"
            
        return ExecuteResult(is_passing, feedback, state)

#     def execute_with_humaneval_testcases(self, name: str, func: str, tests: str, timeout: int = 5):
#         # Extract assert statements from the tests
#         assert_statements = extract_assert_statements(tests)
#         # print(tests)
#         # print(assert_statements)
        
#         # Run the tests and collect the results
#         success_tests = []
#         failed_tests = []
#         is_passing = True
        
#         for test in assert_statements:
#             # Replace the function name with 'candidate'
#             #print("test: ", test)
#             test_with_candidate = re.sub(rf'\bassert\s+{name}\(', 'assert candidate(', test)
#             #print("test with candidate: ", test_with_candidate)
#             # Create the check function
#             check_func = convert_assert_to_check_test(test_with_candidate)
            
#             # Construct the full code to execute
#             code = f"""{func}

# {check_func}

# check({name})
#     """
#             #print(code)
#             try:
#                 # Execute the code
#                 function_with_timeout(exec, (code, globals()), timeout)
                
#                 success_tests.append(test)
#             except Exception as e:
#                 failed_tests.append(f"{test} # output: {str(e)}")
#                 is_passing = False
        
#         # Format the feedback
#         feedback = "Tests passed:"
#         for test in success_tests:
#             feedback += f"\n{test}"
        
#         feedback += "\n\nTests failed:"
#         for test in failed_tests:
#             feedback += f"\n{test}"
        
#         # Create the state tuple
#         state = [test in success_tests for test in assert_statements]
        
#         # Return the result
            
#         return ExecuteResult(is_passing, feedback, state)
    
    
#     def execute_with_humanevalplus_testcases(self, name: str, func: str, tests: str, timeout: int = 5):
#         # Extract assert statements from the tests
#         # print(tests)
#         # exit(0)
#         assert_statements = extract_assert_statements_humanevalplus(tests)
#         # print(tests)
#         # Run the tests and collect the results
#         success_tests = []
#         failed_tests = []
#         is_passing = True
        
#         for test in assert_statements:
#             # Replace the function name with 'candidate'
#             #print("test: ", test)
#             test_with_candidate = re.sub(rf'\bassert\s+{name}\(', 'assert candidate(', test)
#             #print("test with candidate: ", test_with_candidate)
#             # Create the check function
#             check_func = convert_assert_to_check_test(test_with_candidate)
            
#             # Construct the full code to execute
#             code = f"""{func}

# {check_func}

# check({name})
#     """
#             #print(code)
#             try:
#                 # Execute the code
#                 function_with_timeout(exec, (code, globals()), timeout)
                
#                 success_tests.append(test)
#             except Exception as e:
#                 failed_tests.append(f"{test} # output: {str(e)}")
#                 is_passing = False
        
#         # Format the feedback
#         feedback = "Tests passed:"
#         for test in success_tests:
#             feedback += f"\n{test}"
        
#         feedback += "\n\nTests failed:"
#         for test in failed_tests:
#             feedback += f"\n{test}"
        
#         # Create the state tuple
#         state = [test in success_tests for test in assert_statements]
        
#         # Return the result
            
#         return ExecuteResult(is_passing, feedback, state)
    
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        code = f"""{func}

{test}

check({name})
    """
        try:

            function_with_timeout(exec, (code, globals()), timeout)

            return True
        except Exception:
            return False

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    
# def extract_assert_statements(tests_str):
#     """Extract assert statements from the tests string."""
#     # Split by newlines and filter for assert statements
#     return [line.strip() for line in tests_str.split('\n') if line.strip().startswith('assert')]

# def extract_assert_statements_humanevalplus(tests_str):
#     """
#     Extract test cases from the HumanEval format that uses the 'assertion' function
#     with inputs and results lists. It converts these into individual assert statements
#     with the actual input and output values substituted.
    
#     Args:
#         tests_str (str): String containing the test code
        
#     Returns:
#         list: List of strings, each representing an assert statement
#     """
#     # Check if this is the HumanEval format with inputs and results lists
#     inputs_match = re.search(r'inputs\s*=\s*(\[.*?\])', tests_str, re.DOTALL)
#     results_match = re.search(r'results\s*=\s*(\[.*?\])', tests_str, re.DOTALL)
    
#     # If not the expected format, fall back to extracting regular assert statements
#     if not inputs_match or not results_match:
#         return [line.strip() for line in tests_str.split('\n') if line.strip().startswith('assert')]
    
#     # Extract import statements
#     import_statements = []
#     for line in tests_str.split('\n'):
#         if line.strip().startswith('import ') or line.strip().startswith('from '):
#             import_statements.append(line.strip())
    
#     # Create a temporary namespace to evaluate the inputs and results
#     namespace = {}
#     for import_stmt in import_statements:
#         exec(import_stmt, namespace)
    
#     # Safely evaluate the inputs and results lists
#     try:
#         exec(f"inputs = {inputs_match.group(1)}", namespace)
#         exec(f"results = {results_match.group(1)}", namespace)
        
#         inputs = namespace['inputs']
#         results = namespace['results']
#     except Exception as e:
#         # If there's an error evaluating the lists, return a generic assertion
#         return [f"assertion(candidate(*inp), exp, 0) for inp, exp in zip(inputs, results)"]
    
#     # Create individual assert statements for each test case
#     assert_statements = []
    
#     for i, (inp, exp) in enumerate(zip(inputs, results)):
#         # Format the input appropriately
#         if isinstance(inp, list):
#             # For list inputs, we unpack them with *
#             input_str = repr(inp)[1:-1]  # Remove the outer brackets
#         else:
#             # For single inputs, no unpacking needed
#             input_str = repr(inp)
        
#         # Format the assertion with the actual values
#         assert_stmt = f"assertion(candidate({input_str}), {repr(exp)}, 0)"
#         assert_statements.append(assert_stmt)
    
#     return assert_statements

# def convert_assert_to_check_test(assert_statement):
#     """
#     Convert a single assert statement into the check/test_check pattern.
    
#     Args:
#         assert_statement (str): The assert statement to convert
        
#     Returns:
#         str: The converted code with check function
#     """
#     # For HumanEval, asserts are typically in the form: assert func_name(args) == expected
#     # We need to extract func_name to replace it with 'candidate'
    
#     if 'candidate' not in assert_statement:
#         # If it's already in the check function format, return it as is
#         return assert_statement
    
#     # Create the check function with the assertion
#     check_function = f"def check(candidate):\n    {assert_statement}"
    
#     return check_function




if __name__ == "__main__":
    pass
    # Test the function
    # func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    # tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    # print(PyExecutor().execute(func, tests, timeout=1))

    check_function_code = """
def check(candidate):
    assert candidate('') == 0
    assert candidate('x') == 1
    assert candidate('asdasnakj') == 9
def test_check():
    check(strlen)
test_check()
"""
    
    assert_statements = extract_assert_statements(check_function_code)
    print("Extracted assert statements:")
    for i, stmt in enumerate(assert_statements, 1):
        print(f"{i}. {stmt}")