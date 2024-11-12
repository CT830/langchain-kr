from urllib import response
from requests import session
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Multi Tuen 챗봇")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


def load_faiss_db(embeddings):
    db_path = "./faiss_1_row_index"
    faiss_db = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    return faiss_db


def load_retriever():
    # 단계 1: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 단계 2: DB 불러오기
    vectorstore = load_faiss_db(embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # my_prompt = load_prompt("prompts/my-rag.yaml", encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant who is an entrepreneur and knows the relevant support programs."
                "Please answer the question in Korean based on the context you are given."
                "Here is the context you should refer to:: {context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Question\n:{question}",
            ),
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 체인 구성
    # retriever_chain = RunnableParallel(
    #     {"context": lambda _: load_retriever(), "question": RunnablePassthrough()}
    # )

    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        # {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("대화내용을 기억하는 챗봇 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []
    retriver = load_retriever()
    chain = create_chain(retriver, model_name="gpt-4o")
    st.session_state["chain"] = chain

if "store" not in st.session_state:
    st.session_state["store"] = {}


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    # selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 세션 ID 를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if "chain" not in st.session_state:
    retriver = load_retriever()
    st.session_state["chain"] = create_chain(retriver, model_name="gpt-4o")


# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        try:
            response = chain.stream(
                # 질문 입력
                {"question": user_input},
                # 세션 ID 기준으로 대화를 기록합니다.
                config={"configurable": {"session_id": session_id}},
            )

            # 사용자의 입력
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)

                # 대화기록을 저장한다.
                add_message("user", user_input)
                add_message("assistant", ai_answer)
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
    else:
        st.error("체인이 초기화 되지 않았습니다.")
