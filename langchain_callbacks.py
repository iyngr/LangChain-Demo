# Standard Output Callback Handler

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = ''

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("What is the capital of {country}?")

chain = LLMChain(llm=llm, prompt=prompt)
chain.run(country='South Africa', callbacks=[handler])

# File Handler

from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from loguru import logger

logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)
llm = OpenAI()
prompt = PromptTemplate.from_template("What is the capital of {country}? ")

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
answer = chain.run(country='Argentina')

logger.info(answer)

# Custom Callbacks

from langchain.callbacks.base import BaseCallbackHandler


class MyHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def on_llm_start(self, serialized: 'Dict[str, Any]', prompts: 'List[str]', *, run_id: 'UUID',
                     parent_run_id: 'Optional[UUID]' = None, tags: 'Optional[List[str]]' = None,
                     metadata: 'Optional[Dict[str, Any]]' = None, **kwargs: 'Any'): print(
        'We are starting of with this prompt -->', prompts)

    def on_llm_end(self, response: 'LLMResult', *, run_id: 'UUID', parent_run_id: 'Optional[UUID]' = None,
                   **kwargs: 'Any'): print('The LLM is done', response)


handler = MyHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("What is the capital of {country}?")

chain = LLMChain(llm=llm, prompt=prompt)
chain.run(country='South Africa', callbacks=[handler])
