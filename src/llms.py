from typing import Any

import openai

from .pipeline import pipeline, Pipeline
from .utils.token_counters import TiktokenTokenCounter


@pipeline
def OpenAIPipeline(
    text: str,
    model: str = "text-davinci-003",
    max_tokens: int = -1, 
    token_counter: Pipeline[str, int] | None = TiktokenTokenCounter(),
    **kwargs: Any
) -> str:
    text = str(text) # Deal with prompts
    if max_tokens < 0:
        max_tokens = 4097 - token_counter(text) - 200 # 4087 is the max tokens for gpt-3
    if token_counter is not None:
        assert token_counter(text) < max_tokens, "Text is too long for the model"
    kwargs["model"] = model
    kwargs["max_tokens"] = max_tokens
    return openai.Completion.create(prompt=text, **kwargs).choices[0].text

@pipeline
def OpenAIChatPipeline(
    text: str,
    initial_system_message: str = "You are a helpful assistant",
    model: str = "gpt-3.5-turbo",
    max_tokens: int = -1, # make dynamic
    token_counter: Pipeline[str, int] | None = TiktokenTokenCounter(),
    **kwargs: Any
) -> str:
    """
    More support will show up in the future. For now it's just a system message. Followed by a user message.
    """
    text = str(text) # Deal with prompts
    if max_tokens < 0:
        max_tokens = 4097 - token_counter(text) - 200 # 4087 is the max tokens for gpt-3
    if token_counter is not None:
        assert token_counter(text) < max_tokens, "Text is too long for the model"
    kwargs["model"] = model
    kwargs["max_tokens"] = max_tokens
    return openai.ChatCompletion.create(messages=[
        {"role": "system", "text": initial_system_message},
        {"role": "user", "text": text},
    ], **kwargs).choices[0].message.content