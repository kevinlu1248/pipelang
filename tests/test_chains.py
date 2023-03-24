import os
from dotenv import load_dotenv
import openai
import requests

from src.llms import OpenAIChatPipeline
from src.prompts import PromptPipeline
from src.chains import MapReduceQAPipeline, MapReduceSummaryPipeline

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def test_prompt_insert_pipeline_simple():
    simple_pipeline = PromptPipeline(template='Say "{text}"?') >> OpenAIChatPipeline(temperature=0)
    assert simple_pipeline("Hello world!").strip() == "Hello world!"

def test_prompt_insert_pipeline():
    simple_pipeline = PromptPipeline(template='Why do developers print "{text}"?') >> OpenAIChatPipeline()
    print(simple_pipeline("Hello world!"))

def test_summary():
    text = requests.get("https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt").text
    pipeline = MapReduceSummaryPipeline()
    print(pipeline(text))

def test_qa():
    text = requests.get("https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt").text
    question = "What did the president say about Justice Breyer?"
    pipeline = MapReduceQAPipeline(question)
    print(pipeline(text))
