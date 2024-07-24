import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
prompt = PromptTemplate.from_template("Complete a {length} story using the given beginning. The genre should be {"
                                      "genre} and the story should have an apt ending. Beginning: {text}")

llm = OpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-instruct", max_tokens=250)
chain = LLMChain(llm=llm, prompt=prompt)

print('\n'.join(
    chain.run({'length': 'short', 'genre': 'horror', 'text': 'Once there was a coder'}).replace('\n', '.').split('.')))
print('\n'.join(
    chain.run({'length': 'short', 'genre': 'rom-com', 'text': 'And the Queen died'}).replace('\n', '.').split('.')))
