from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# step1: setup env
load_dotenv()

# step2: setup llm
llm = ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)
