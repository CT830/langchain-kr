import streamlit as st
from langchain_core.messages import ChatMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def load_faiss_db(embeddings):
    db_path = "./faiss_index"
    faiss_db = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    return faiss_db


def load_retriever():

    # 단계 1: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 단계 2: DB 불러오기
    # 벡터스토어를 생성합니다.
    vectorstore = load_faiss_db(embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def embed_context():
    # CSV 파일 경로
    loader = CSVLoader(
        file_path="./data/announcement.csv",
        csv_args={
            "delimiter": ",",  # 구분자
            "quotechar": '"',  # 인용 부호 문자
            "fieldnames": [
                "Sequence",  # 순번
                "Registration number",  # 공고번호
                "Announcement name",  # 공고명
                "Support areas",  # 지원분야
                "Region",  # 지역
                "Target",  # 지원대상
                "Target age",  # 대상연령
                "Application period",  # 접수기간
                "Entrepreneurial history",  # 업력
                "Institution name",  # 기관명
                "Organization classification",  # 기관구분 : 공공, 민간, 교육
                "Department in charge",  # 담당부서
                "Announcement number",  # 공고 제 호
                "Announcement contents",  # 공고내용
                "Registration date",  # 공고등록일
                "Announcement registrar name",  # 공고 기업명
                "How to apply work-in",  # 신청방법 : 방문
                "How to apply By mail",  # 신청방법 : 우편
                "How to apply By Fax",  # 신청방법 : FAX
                "How to apply By email",  # 신청방법 : email
                "How to apply online",  # 신청방법 : 온라인
                "How to apply other",  # 신청방법 : 기타
                "Who to apply for",  # 신청대상
                "Excluded from application",  # 제외대상
                "Summary",  # 공고명 + 공고내용
            ],  # 필드 이름
        },
    )

    # 데이터 로드
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def create_chain(retriever, model_name="gpt-4o"):

    prompt = load_prompt("./prompts/my-rag.yaml", encoding="utf-8")

    llm = ChatOpenAI(model_name=model_name, temperature=0)

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

st.title("My ChatGPT")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []
    # retriever = embed_context()
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
