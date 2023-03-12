import os
from dotenv import load_dotenv
import openai
from llms import OpenAIPipeline

from prompt import PromptInsertPipeline

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def test_prompt_insert_pipeline():
    simple_pipeline = PromptInsertPipeline(
            prompt='Why do programmers always print "{text}"?', key="text"
        ) >> OpenAIPipeline()
    print(simple_pipeline("Hello world!"))

