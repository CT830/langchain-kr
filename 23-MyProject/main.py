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


def create_chain(retriever, model_name="gpt-4o"):
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 Startup 전문가이며 친절한 AI 어시스턴트"),
            ("user", "#Question:\n{question}"),
        ]
    )
    prompt = load_prompt("./prompts/my-rag.yaml", encoding="utf-8")
    # 프롬프트 템플릿 직접 정의
    # template = """당신은 startup 전문가로 정부 및 민간의 startup을 지원하는 프로그램에 대해서 잘 알고있습니다.
    #     다음 컨텍스트를 바탕으로 친절하게 답변해주세요.

    #     context: {context}
    #     question: {question}

    #     answer:"""

    # template = """You are a startup expert and are familiar with government and private programs that support startups.
    #     Using the following context, answer the question including Support aresa, Region, Announcement name, Announcement contents,
    #     Target, Target age, Entrepreneural history, How to apply work-in, How to apply By mail, How to apply By Fax,
    #     How to apply By email, How to apply online, How to apply other, Who to apply for, Exclueded from applicatioin
    #     and the application period, or answer that you don't know if it is not in context.
    #     Please answer in Korean.

    #     context: {context}
    #     question: {question}

    #     answer:"""

    # template = """You are a startup expert and are familiar with government and private programs that support startups.
    #     Using the following context, answer the question in the following format.

    #     Overview of the program : describe it using Summary\n
    #      - Announcement : \n
    #      - Announcement contents: \n
    #      - Support area : \n
    #      - Target : \n
    #      - Target age : \n
    #      - Region : \n
    #      - Entrepreneural history : \n
    #      - How to apply work-in: \n
    #      - How to apply By mail: \n
    #      - How to apply By fax: \n
    #      - How to apply By email: \n
    #      - How to apply By online: \n
    #      - Who to apply for : \n
    #      - Exceluded from application : \n
    #      - Application period : \n

    #        or answer that you don't know if it is not in context. Please answer in Korean.

    #     context: {context}
    #     question: {question}

    #     answer:"""

    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # GPT
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # output parser
    output_parser = StrOutputParser()

    # create chain
    chain = (
        {
            "context": lambda x: format_docs(
                retriever.get_relevant_documents(x["question"])
            ),
            "question": RunnablePassthrough(),
            # "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | output_parser
    )
    return chain


# API KEY 정보로드
load_dotenv()

logging.langsmith("CH23-MyProject")

st.title("My ChatGPT")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = ""

# 이전 대화 기록
print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요")
if user_input:
    # 웹에 대화를 출력
    st.chat_message("user").write(user_input)

    retriever = load_retriever()
    chain = create_chain(retriever)

    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    ai_answer = chain.invoke({"question": user_input})

    # 대화기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
