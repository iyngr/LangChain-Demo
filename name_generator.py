import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
prompt = PromptTemplate.from_template("Give {number} names for a {domain}  startup?")
llm = OpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-instruct")
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({'number': '5', 'domain': 'cooking'}))
print(chain.run({'number': '2', 'domain': 'AI'}))
