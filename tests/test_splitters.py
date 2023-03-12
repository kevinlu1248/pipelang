import requests
from src.splitters import RecursiveTextSplitter

def test_recursive_splitter():
    # text = "Test.\n\nTest."
    text = "Hello my name is Joe. What is your name?"
    splitter_pipeline = RecursiveTextSplitter(max_tokens = 5)
    print(splitter_pipeline(text))

def test_recursive_splitter_large():
    text = requests.get("https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt").text
    splitter_pipeline = RecursiveTextSplitter(max_tokens = 1024)
    splitted = splitter_pipeline(text)
    for text in splitted:
        print(text.count(' ') + 1) 

