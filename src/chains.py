# Might need vectorstore: problem for later

from functools import reduce
from pipeline import JoinerLike, LLMLike, Pipeline, SplitterLike, pipeline
from splitters import RecursiveTextSplitter


@pipeline
def RefinePipeline(
    text: str, 
    initializer: LLMLike,
    refiner: Pipeline[tuple[str, str], str],
    splitter: SplitterLike = RecursiveTextSplitter()
) -> str:
    first, *rest = splitter(text)
    return reduce(refiner, rest, initializer(first))

@pipeline
def MapReducePipeline(
    text: str, 
    mapper: LLMLike, 
    reducer: JoinerLike, 
    splitter: SplitterLike = RecursiveTextSplitter(),
) -> str:
    return reducer(text, map(mapper, splitter(text)))
