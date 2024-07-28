# ConversationalBufferMemory


from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = ''

template = """You are a chatbot having a conversation with a human. 
{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI()
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)
llm_chain.predict(human_input="Hi there my friend")

# ConversationSummaryMemory

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context({"input": "hi"}, {"output": "whats up"})
llm = OpenAI(temperature=0)
chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

chat.predict(input='Hey dude !! do you know what happened to Nandu?')
chat.predict(input='He met with a car accident yesterday. Suggest me some hospitals for a good treatment')
