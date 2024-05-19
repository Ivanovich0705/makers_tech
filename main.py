from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
import chainlit as cl
from langchain import SQLDatabase


HUGGINGFACEHUB_API_TOKEN = "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"

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 1.0, "max_new_tokens": 250})

db = SQLDatabase.from_uri("mysql+pymysql://root:root@localhost:3306/makers_tech")
agent_executor = create_sql_agent(llm, db=db, verbose=True)


template = """You are a tech shop assistant that will help customers know the stock of products, what products we 
have avaliable depending their needs, prices, features. Your expertise is exclusively in providing information and 
advice about anything related to technology products. 
Question: {question}
Answer:"""


@cl.on_chat_start
def main():
    conversation_buf = ConversationChain(
        llm=agent_executor,
        # We set a very low max_token_limit for the purposes of testing.
        memory=ConversationSummaryBufferMemory(llm=llm),
        verbose=True,
    )
    cl.user_session.set("conversation_buf", conversation_buf)


@cl.on_message
async def main(message):
    conversation_buf = cl.user_session.get("conversation_buf")

    res = conversation_buf.invoke(input=message.content)
    await cl.Message(content=res).send()