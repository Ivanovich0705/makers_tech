from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
import os
from dotenv import load_dotenv
import chainlit as cl


HUGGINGFACEHUB_API_TOKEN = "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"

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 1.0, "max_new_tokens": 250})

@cl.on_chat_start
def main():
    conversation_buf = ConversationChain(
        llm=llm,
        # We set a very low max_token_limit for the purposes of testing.
        memory=ConversationSummaryBufferMemory(llm=llm),
        verbose=True,
    )
    cl.user_session.set("conversation_buf", conversation_buf)


@cl.on_message
async def main(message):
    conversation_buf = cl.user_session.get("conversation_buf")

    res = conversation_buf.predict(input=message.content)
    await cl.Message(content=res).send()