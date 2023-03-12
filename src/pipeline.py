import functools
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel

# We need a preparation system for expensive operations

InputType = TypeVar("InputType") # default to str in Python 3.12
OutputType = TypeVar("OutputType") # default to str in Python 3.12
ArgsType = TypeVar("ArgsType")
KwargsType = TypeVar("KwargsType")

class Pipeline(Generic[InputType, OutputType]):
    def __init__(self, func: Callable[[InputType], OutputType], *args: ArgsType, **kwargs: KwargsType) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, text: InputType) -> OutputType:
        return self.func(text, *self.args, **self.kwargs)

# Add docstrings and stuff
def chain(self, other: Pipeline[InputType, OutputType]) -> Pipeline[InputType, OutputType]:
    """Chain two pipelines together."""
    return Pipeline(lambda text: other(self(text)))

Pipeline.__rshift__ = chain
Pipeline.__lshift__ = lambda self, other: chain(other, self)

def pipe(self, other: Pipeline[OutputType, OutputType]) -> Pipeline[InputType, OutputType]:
    """Pipe two pipelines together."""
    return Pipeline(lambda text: map(other, self(text)))

Pipeline.__or__ = pipe

class PipelineFactory(Generic[InputType, OutputType]):
    def __init__(self, func: Callable[[InputType], OutputType]) -> None:
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Pipeline[InputType, OutputType]:
        return Pipeline(self.func, *args, **kwargs)

def pipeline(func: Callable[[InputType], OutputType]) -> PipelineFactory[InputType, OutputType]:
    return PipelineFactory(func)

LLMLike = Pipeline[str, str]
SplitterLike = Pipeline[str, list[str]]
JoinerLike = Pipeline[list[str], str]
