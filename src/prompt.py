# TODO: Add default options
from typing import Any 

from .pipeline import pipeline

@pipeline
def PromptInsertPipeline(
    text: str, 
    prompt_template: str, 
    key: str,
    **kwargs: Any
) -> str:
    return prompt_template.format(**{key: text})

@pipeline
def PromptInsertMultiplePipeline(
    data: dict[str, str],
    prompt_template: str, 
    **kwargs: Any
) -> str:
    return prompt_template.format(data)
