import streamlit as st
from langchain_core.messages import ChatMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging
from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os, time
import warnings


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def load_faiss_db(embeddings):
    db_path = "./faiss_hfe_index"
    print("load_local")
    faiss_db = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    return faiss_db


def load_retriever():

    # 단계 1: 임베딩(Embedding) 생성
    #        HuggingFace Embedding Local 사용(./cache 디렉토리에 저장)
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    model_name = "intfloat/multilingual-e5-large-instruct"

    start_time = time.perf_counter()
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "mps"},  # cuda, cpu, mps
        encode_kwargs={"normalize_embeddings": True},
    )
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    print(f"embedding time : {exec_time:.6f} sec")

    start_time = time.time()
    # 단계 2: DB 불러오기
    # 벡터스토어를 생성합니다.
    vectorstore = load_faiss_db(hf_embeddings)
    end_time = time.time()
    print(f"load time : {exec_time:.6f} sec")

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


def create_chain(retriever, model_name="gpt-4o"):

    prompt = load_prompt("./prompts/my-rag.yaml", encoding="utf-8")

    llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0-Q4_1")
    #llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# API KEY 정보로드
load_dotenv()

logging.langsmith("CH23-MyProject")

# 경고 무시
warnings.filterwarnings("ignore")

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["HF_HOME"] = "./cache/"
# HugginFace tokenizers warning 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("My ChatGPT")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []
    retriver = load_retriever()
    chain = create_chain(retriver, model_name="gpt-4o")
    st.session_state["chain"] = chain

with st.sidebar:
    clear_btn = st.button("대화 초기화")


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = ""

# 이전 대화 기록
print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요")

if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
