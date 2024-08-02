from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl
import torch
from ctransformers import AutoModelForCausalLM

DB_FAISS_PATH = 'vectorstore/db_faiss'

prompt_template = """Use of followig piece of information to answer user question.
If you don't know the answer , just say that you don't know

Context:{context}
Question:{question}

Only return the helpful and nothing else:

Helpful answer:
"""

def set_prompt():
    prompt = PromptTemplate(template = prompt_template , input_variables=['context' , 'question'])
    return prompt

def load_llm():
    llm = CTransformers(model = 'llama-2-7b-chat.ggmlv3.q8_0.bin' ,model_type = 'llama', max_new_tokens = 512 , temperature = 0.5)
    return llm

def retrieval_qa_chain(llm , prompt , db):
    qa_chain = RetrievalQA.from_chain_type(llm = llm ,  retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,chain_type_kwargs={'prompt': prompt})
    
    return qa_chain


    
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings , allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa    


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response




#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()