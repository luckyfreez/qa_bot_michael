"""Python file to serve as the frontend"""
import streamlit as st
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
import faiss
from langchain import OpenAI, PromptTemplate
from langchain.chains import VectorDBQAWithSourcesChain, RetrievalQA
import pickle

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
# chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever(),
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# From here down is all the StreamLit UI.
st.set_page_config(page_title="内部AI机器人", page_icon=":robot:")
st.header("西美内部AI机器人")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "你好，请问如何创建交货单？", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"query": user_input})
    output = f"Answer: {result['result']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
