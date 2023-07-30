"""Ask a question to the notion database."""
import faiss
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

template = """你是西美公司的内部wiki机器人，以下文档是从公司内部wiki里面截取的一部分信息。请用以下信息来回答问题。
如果你不知道就说不知道，不要编造内容。

===以下为参考信息===
{context}
======

请回答这个问题： 
{question}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

store.index = index
chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever(),
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
result = chain({"query": args.question})
print(f"Answer: {result['result']}")
