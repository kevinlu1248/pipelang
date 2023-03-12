import tiktoken

from ..pipeline import pipeline

@pipeline
def SimpleTokenCounter(text: str, deliminator: str | None = None) -> int:
    """
    Placing this here for convenience. However, it is not recommended to use this.
    """
    return len(text.split(deliminator))

@pipeline
def TiktokenTokenCounter(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    cl100k_base for ChatGPT
    p50k_base for code and davinci 2/3
    r50k_base for davinci 1
    """
    encoding = tiktoken.get_encoding(encoding_name) # might be expensive
    return len(encoding.encode(text))