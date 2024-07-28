from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']
youtube = YouTubeSearchTool()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    Tool(
        name="youtube",
        func=youtube.run,
        description="Helps in getting youtube videos",
    ),
    Tool(
        name="wiki",
        func=wiki.run,
        description="Useful to search about a popular entity",
    )
]
prefix = "Answer the question asked. You have access to the following tools:"

suffix = "Begin!" \
         "{chat_history}" \
         "Question: {input}" \
         "{agent_scratchpad}"

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"], )
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-instruct", max_tokens=250, temperature=0),prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory)
agent_chain.run(input="Tell me something about Hugh Jackman. Also, get me the link to one of his movies from YouTube")
