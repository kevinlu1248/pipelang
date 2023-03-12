from .token_counters import TiktokenTokenCounter
from ..pipeline import Pipeline, pipeline

@pipeline
def RecursiveTextSplitter(
    text: str, 
    deliminators: list[str] = ["\n\n", "\n", ".", "?", "!", " "], 
    token_counter: Pipeline[str, int] = TiktokenTokenCounter(),
    max_tokens: int = 1024
) -> list[str]:
    """
    Splits recursively like langchain. Also can be optimized but not the bottleneck
    TODO: make more robust like the one by langchain by adding overlap
    """
    max_tokens -= 1 # for deliminators
    current_texts = [text]
    for splitter in deliminators:
        next_texts = []
        did_split = False
        for current_text in current_texts:
            if token_counter(current_text) > max_tokens:
                split_text = current_text.split(splitter)
                split_text = [_text + splitter for _text in split_text[:-1]] + [split_text[-1]]
                next_texts.extend(split_text)
                did_split = True
            else:
                next_texts.append(current_text)
        current_texts = next_texts
        if not did_split:
            break
    for current_text in current_texts:
        if token_counter(current_text) > max_tokens:
            raise ValueError("Text is too long")
    new_texts = []
    last_text_size = -1
    for current_text in current_texts:
        if last_text_size >= 0 and token_counter(current_text) + last_text_size <= max_tokens:
            new_texts[-1] += current_text
            last_text_size += token_counter(current_text)
        else:
            new_texts.append(current_text)
            last_text_size = token_counter(current_text)
    for current_text in current_texts:
        if token_counter(current_text) > max_tokens:
            print(token_counter(current_text), max_tokens)
            raise ValueError("Text is too long")
    while new_texts[-1] == "":
        new_texts.pop()
    return new_texts 
