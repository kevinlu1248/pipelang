# pipelang (in progress / proof of concept)
Pipelines for large language models. Langchain wrapper for for extensibility, versatility, and transparency and without the complexity using functional components.

Ever try out Langchain and love the features and how everything works out of the box but simply find that there is little room for extensibility, and find it really difficult to modify? Ever wonder how to modify the prompts for a map-reduce QA pipeline? Well that's how I felt and why I built Pipelang.

For the full story, check out "Story".

## Installation

Install with 
```python
pip install pipelang
```

## Quickstart 

Write a quick map-reduce QA pipeline with clarity and transparency:

```python
from src.llms import OpenAIPipeline
from src.prompts import Prompt
from src.utils.joiners import SimpleJoinerPipeline
from src.utils.splitters import RecursiveTextSplitter

text = requests.get("https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt").text
question = "What did the president say about Justice Breyer?"

mapper_prompt: Prompt = Prompt.from_file('src/prompts/map_reduce_qa/map_langchain.txt')
reducer_prompt: Prompt = Prompt.from_file('src/prompts/map_reduce_qa/reduce_langchain.txt')

mapper_prompt_pipeline = mapper_prompt.fill(question, "question").fill_pipeline("context")
reducer_prompt_pipeline = reducer_prompt.fill(question, "question").fill_pipeline("summaries")

map_reduce_pipeline = (RecursiveTextSplitter() | (mapper_prompt_pipeline >> OpenAIPipeline())) \
    >> SimpleJoinerPipeline() >> reducer_prompt_pipeline >> OpenAIPipeline()
print(map_reduce_pipeline(text))
```

Or simply use the built-in MapReduceQAPipeline:

```python
from src.chains import MapReduceQAPipeline

map_reduce_pipeline = MapReduceQAPipeline(question)
```

Features:
* Base Pipeline system
* LLMs (only OpenAI so far)
* Prompts (with all the langchain prompts)
* Chains and Splitters (All the ones from langchain)

Future:
* Thorough type-checking
* Asynchronous calls
* Memory
* Flowchart generation / Computation graph
* Pipeline factories
* Tracing
* Making langchain an optional dependency

Attributions:
* Langchain, which taught me a lot on modern techniques in working with LLM's such as chains and agents, as well as the framework
* React, which taught me FP and what a simple, powerful and extensible API looks like