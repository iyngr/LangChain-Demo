# Comma Separated List Output Parser

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

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
