# Comma Separated List Output Parser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

from langchain_openai import ChatOpenAI

api_key = ""
os.environ['OPENAI_API_KEY'] = api_key

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(template="Suggest some names for my {subject} startup.\n {format_instructions}",
                        input_variables=["subject"],
                        partial_variables={"format_instructions": format_instructions}
                        )
model = OpenAI(temperature=0)
_input = prompt.format(subject="Mobile")
output = model(_input)
output_parser.parse(output)

# Using LangChain LCEL

prompt = PromptTemplate.from_template(
    "Answer the following question: {question}"
)

parser = CommaSeparatedListOutputParser()
model = OpenAI(temperature=0)
chain = prompt | model | parser
list(chain.invoke({"question": "Suggest some names for baby"}))

# Custom Output Parsers

response_schemas = [
    ResponseSchema(name="Monument", description="The monument mentioned in the answer."),
    ResponseSchema(name="city", description="the city in which this monument is present."),
    ResponseSchema(name="architect", description="the architect of the monument."), ]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="answer the users question as best as possible.\n {format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)
model = OpenAI(temperature=0)

_input = prompt.format_prompt(question="Give example of a monument in North America with details about it")
output = model(_input.to_string())
output_parser.parse(output)


# Magic Output Fixer

class state(BaseModel):
    state: str = Field(description="name of a Indian state")
    cities: List[str] = Field(description="Indian cities in that state")


parser = PydanticOutputParser(pydantic_object=state)

format_instructions = parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible and generate output. \n {format_instructions}\n{question}",
    input_variables=["question"], partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)
query = "Choose a random Indian state and pen down some cities"

_input = prompt.format_prompt(question=query)
output = model(_input.to_string())

try:
    print(parser.parse(output))
except:
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
print(new_parser.parse(output))
