from langfuse import get_client
from langfuse.langchain import CallbackHandler

from common import llm

# step1: input
customer_ticket = """
Hello, I noticed an unrecognized charge of $49.99 on my credit card statement
from your company yesterday. I did not authorize this transaction.
Please help me process a refund immediately.
"""

# step2: execute llm call
print("Processing customer support ticket...")
response = llm.invoke(
    f"You are an AI support assistant. Summarize this ticket in one sentence:\n{customer_ticket}",
    config={"callbacks": [CallbackHandler()]},
)

print("\n--- AI Summary ---")
print(response.content)

get_client().flush()
