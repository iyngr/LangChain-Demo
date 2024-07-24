import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
prompt = PromptTemplate.from_template("Preprocess the given text by following the given steps in sequence. Follow only "
                                      "those steps that have a yes against them. Remove Number:{number},"
                                      "Remove punctuations : {punc} ,Word stemming : {stem}.Output just the "
                                      "preprocessed text. Text : {text}")

llm = OpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-instruct")
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({'text': 'Hey!! I got 12 out of 20 in Swimming', 'number': 'yes', 'punc': 'yes', 'stem': 'no'}))
print(chain.run({'text': '22 13B is my flat no. Rohit will be joining us for the party', 'number': 'yes', 'punc': 'no',
                 'stem': 'yes'}))
