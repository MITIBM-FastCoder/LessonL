from abc import ABC, abstractmethod

class AbstractLLMAgent(ABC):
    """
    Abstract base class for an agent that interacts with a Large Language Model (LLM) using prompts.
    """
    
    def __init__(self, llm):
        """
        Initialize the agent with a reference to the LLM.

        Args:
            llm: An instance of the LLM to be used for inference.
        """
        self.llm = llm
    
    @abstractmethod
    def generate_prompt(self, context):
        """
        Generate a prompt based on the given context.

        Args:
            context: A dictionary or other structured data containing information
                     necessary to construct the prompt.

        Returns:
            str: The generated prompt.
        """
        pass
    
    @abstractmethod
    def process_response(self, response):
        """
        Process the response from the LLM.

        Args:
            response: The raw response from the LLM.

        Returns:
            The processed output in a format suitable for the agent's purpose.
        """
        pass
    
    @abstractmethod
    def infer(self, prompt):
        """
        Perform inference by using a prompt, querying the LLM, and processing the response.

        Args:
            prompt: the prompt that used to inference LLM

        Returns:
            The processed result of the LLM's response.
        """
        # prompt = self.generate_prompt(context)
        # raw_response = self.llm.query(prompt)
        # return self.process_response(raw_response)
        pass