# Might need vectorstore: problem for later
# Figure out async

from functools import reduce
from typing import Callable

from src.filters import FilterLike, NoFilter

from .utils.joiners import SimpleJoinerPipeline
from .pipeline import JoinerLike, LLMLike, Pipeline, SplitterLike, pipeline
from .prompts import Prompt, PromptPipeline
from .utils.splitters import RecursiveTextSplitter
from .llms import OpenAIPipeline

@pipeline
def MapReducePipeline(
    text: str, 
    mapper: LLMLike | Callable[[str], LLMLike],
    reducer: JoinerLike | Callable[[str], JoinerLike],
    splitter: SplitterLike = RecursiveTextSplitter(),
    filter: FilterLike = NoFilter()
) -> str:
    """
    Technically not very robust as it can still break the limit. Should be called recursively instead.
    """
    if not isinstance(mapper, Pipeline):
        mapper = mapper(text)
    if not isinstance(reducer, Pipeline):
        reducer = reducer(text)
    mapper: LLMLike
    reducer: JoinerLike
    pipeline = ((splitter >> filter) | mapper) >> reducer
    return pipeline(text)

def MapReduceSummaryPipeline(
    prompt: LLMLike = PromptPipeline(template_path='src/prompts/summarize_langchain.txt'), 
    llm: LLMLike = OpenAIPipeline(max_tokens=-1),
    joiner: JoinerLike = SimpleJoinerPipeline(),
    **kwargs
):
    summarize_pipeline = prompt >> llm
    return MapReducePipeline(
        summarize_pipeline,
        joiner >> summarize_pipeline,
        **kwargs
    )

def MapReduceQAPipeline(
    question: str,
    mapper_prompt: Prompt = Prompt.from_file('src/prompts/map_reduce_qa/map_langchain.txt'),
    reducer_prompt: Prompt = Prompt.from_file('src/prompts/map_reduce_qa/reduce_langchain.txt'),
    joiner_pipeline: LLMLike = SimpleJoinerPipeline(),
    llm: LLMLike = OpenAIPipeline(max_tokens=-1),
    **kwargs
):
    return MapReducePipeline(
        mapper_prompt.fill(question, "question").fill_pipeline("context") >> llm,
        joiner_pipeline >> reducer_prompt.fill(question, "question").fill_pipeline("summaries") >> llm,
        **kwargs
    )

@pipeline
def RefinePipeline(
    text: str, 
    initializer: LLMLike,
    refiner: Pipeline[tuple[str, str], str],
    splitter: SplitterLike = RecursiveTextSplitter()
) -> str:
    first, *rest = splitter(text)
    return reduce(refiner, rest, initializer(first))
