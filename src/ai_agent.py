import textwrap

import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


# ----------------------------
# Pydantic Model
# ----------------------------
class ResearchResponse(BaseModel):
    topic: str = Field(..., description="The topic of the research paper")
    summary: str = Field(..., description="A summary of the research paper")
    sources: list[str] = Field(..., description="A list of sources used in the research paper")
    tools_used: list[str] = Field(..., description="A list of tools used in the research paper")


# ----------------------------
# LLM + Parser Setup
# ----------------------------
llm = ChatAnthropic(
    model_name="claude-sonnet-4-5-20250929",
    temperature=0,
    timeout=None,
    max_retries=2
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

SYSTEM_PROMPT = """
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output of this format and provide no other text
{format_instructions}
"""

prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser


# ----------------------------
# Helpers
# ----------------------------
def pretty_print_wrapped(label: str, text: str, width: int = 120):
    """Utility to print wrapped key/value lines for output."""
    indent = f"{label}: "
    print(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=" " * len(indent)
        )
    )
    print()  # blank line


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":

    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the meaning of life?"
    )

    max_width = 120

    pretty_print_wrapped("Original Question", question, width=max_width)

    output: ResearchResponse = chain.invoke({"question": question})

    for field_name, field_value in output.model_dump().items():
        label = field_name.replace("_", " ").title()
        if isinstance(field_value, list):
            field_value = ", ".join(field_value)
        pretty_print_wrapped(label, field_value, width=max_width)
