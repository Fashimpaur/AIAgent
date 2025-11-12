from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model_name="claude-sonnet-4-5-20250929",
                    temperature=0,
                    timeout=None,
                    max_retries=2,
                    stop=None)
response = llm.invoke("Tell me a joke")
print(response)
