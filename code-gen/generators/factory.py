from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .go_generate import GoGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, GPT4omini, GPT4o, StarChat, GPTDavinci, Qwen14bVLLM, Qwen7bVLLM, Deepseek7bVLLM


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    elif lang == "go" or lang == "golang":
        return GoGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo-0613":
        return GPT35()
    elif model_name == "gpt-4o-mini":
        return GPT4omini()
    elif model_name == "gpt-4o":
        return GPT4o()
    elif model_name == "starchat":
        return StarChat()
    elif model_name == "Qwen/Qwen2.5-Coder-14B-Instruct":
        return Qwen14bVLLM(model_name)
    elif model_name == "Qwen/Qwen2.5-Coder-7B-Instruct":
        return Qwen7bVLLM(model_name)
    elif model_name == "deepseek-ai/deepseek-coder-7b-instruct-v1.5":
        return Deepseek7bVLLM(model_name)
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
