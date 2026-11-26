from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

llm = ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)

customer_ticket = """
Hello, I noticed an unrecognized charge of $49.99 on my credit card statement 
from your company yesterday. I did not authorize this transaction. 
Please help me process a refund immediately.
"""

print("Processing customer support ticket...")

response = llm.invoke(
    f"You are an AI support assistant. Summarize this ticket in one sentence:\\n{customer_ticket}",
    config={"callbacks": [langfuse_handler]},
)

# 6\. Output the result to the console
print("\\n--- AI Summary ---")
print(response.content)
