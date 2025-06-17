import streamlit as st
import openai
import pandas as pd
import os  # 환경변수 사용을 위해 필요
import sys, re
import requests

# langchain core
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

# langchain community
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

# 페이지 레이아웃을 wide로 지정
st.set_page_config(layout="wide")

# CSS 적용 (버튼에 전체 적용)
st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #FF6347;
        float:right;
    }
    </style>
    """, unsafe_allow_html=True)

# Final Project 머릿말 (왼쪽 상단)
st.markdown(
    "<div style='position: absolute; top: 20px; right: 20px; font-size: 14px; color: gray;'>"
    "<b>Final Project — <i>Understanding and Application of Foundation Models</i></b><br>"
    "</div>",
    unsafe_allow_html=True
)
st.text("\n")

# Title - Sub Title
st.markdown("<h1 style='text-align: center; color: black; padding-top: 40px;'>ChatGPT Code Rule Checker (using GPT-4o)</h1>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; color: gray;'>Master of Computer Software Engineering, Yonsei University</h5>"
    "<h5 style='text-align: center; color: gray;'>Taesu Kim (2024451104)</h5>",
    unsafe_allow_html=True
)

st.text("\n")
st.text("\n")
st.text("\n")
# Create two columns
col1, col2 = st.columns(2)

# Use the columns like normal st calls
col1.subheader("[ 데이터 입력 ]")
col1.markdown("##### 1. 작성한 소스코드 검사를 위한 룰셋을 업로드하세요. (샘플 : java)")

# 1) get_excel_chunks 메소드
# 설명 : to_string으로 변환된 엑셀 데이터를 chunks 로 쪼갬
def get_excel_chunks(data_st):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(data_st)
    return chunks

# 2) get_vectorstore 메소드
# 설명 : excel chunks를 벡터화해서 FAISS라는 검색 라이브러리에 저장함 (쉽게 말해, 쪼갠 chunks 들을 vector store 에 저장한다고 보면 됌)
def get_vectorstore(excel_text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(texts=excel_text_chunks, embedding=embeddings)
    return vectorstore

# 3) get_conversation_chain 메소드 (핵심 로직, gpt-4o 사용)
# 설명 : 사용자와 gpt 챗봇 간의 ConversationalRetrievalChain 을 생성함
#       이 대화체인을 통해, 사용자의 질문에 대한 적절한 응답을 생성하기 위해 언어모델, 메모리, 벡터 검색기를 사용함
# 참고 : Retriever는 사용자의 질문이나 주제에 대해 미리 학습된 데이터에서 가장 관련이 있는 정보를 찾아주는 역할을 함. 이걸 쓰기 위해서 벡터db에 저장해야 함
def get_conversation_chain(vectorstore):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, streaming=True, callbacks=callbacks)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# 4) run 메소드 (실제 최초 동작부)
def run():
    uploaded_file = col1.file_uploader("Choose a Excel file", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        with col1.expander("적용된 룰셋을 상세히 보기"):
            keyword = st.text_input("검색어를 입력하세요")
            if keyword:
                search_result = data[data['RULE_NAME'].astype(str).str.lower().str.contains(keyword.lower())]
                st.write(search_result)
            else:
                st.write(data)
            data_st = data.to_string()
            excel_text_chunks = get_excel_chunks(data_st)
            vectorstore = get_vectorstore(excel_text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        col1.write("파일을 업로드 해주세요.")

if __name__ == '__main__':
    run()

col1.text("\n")
col1.text("\n")
col1.text("\n")
col1.markdown("##### 2. CPSP 검사를 위한 소스코드를 입력하세요.")
col2.subheader("[ 결과 출력 ]")

user_input = col1.text_area("Please enter your text here")

# 5) handle_userinput 메소드
# 설명 : 사용자가 입력한 소스코드를 라인단위로 읽어 string 연산 한 결과를 가져다가, 위에서 구현한 대화체인에게 질의함
#       질의한 결과를 st.session_state.displayed_chat_history에 append 함
def handle_userinput(check_datas):
    response = st.session_state.conversation({'question': check_datas})
    st.session_state.chat_history = response['chat_history']
    st.session_state.displayed_chat_history = []
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:
            st.session_state.displayed_chat_history.append(message.content)

# Slack 메시지만 따로 뽑기 위한 헬퍼 함수
def extract_slack_message(full_response):
    """
    GPT의 전체 응답에서 Slack 메시지용 포맷만 추출한다.
    Slack 메시지는 '*✅ 코드 룰셋 검사 결과*'로 시작한다고 가정.
    """
    lines = full_response.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("*✅ 코드 룰셋 검사 결과*"):
            start_idx = i
            break
    if start_idx is not None:
        return "\n".join(lines[start_idx:])
    else:
        return "⚠️ Slack 메시지 포맷을 찾을 수 없습니다."

# Slack 알림 전송 함수
# 입력: message (str) - Slack으로 전송할 메시지
# 비고: Streamlit secrets에 SLACK_WEBHOOK_URL이 등록되어 있어야 함
def send_to_slack(message):
    webhook_url = st.secrets["SLACK_WEBHOOK_URL"]
    payload = {
        "text": f"{message}"
    }
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        st.error("Slack 전송 실패: " + response.text)

if col1.button("검사시작", key="button"):
    with st.spinner("검사 중입니다..."):
        check_data = user_input
        check_datas = ''
        line_all = ''
        lines = check_data.splitlines()
        for line in lines:
            line_all = line_all + line
            print('line: '+line_all)

        # 별도 질의를 사용자가 입력하지 않기 위함
        with open('prompt/userQuery', 'r') as f:
            lines = f.readlines()
            user_query = " ".join(line.strip() for line in lines)

        check_datas = line_all+ '\n' + user_query
        handle_userinput(check_datas)
        st.session_state.previous_question = line_all

        # HTML 태그 제거 및 결과 정리
        clearer = re.compile('<.*?>')
        if 'displayed_chat_history' in st.session_state:
            full_result = []
            for message in st.session_state.displayed_chat_history:
                rmT = re.sub(clearer, '', message)
                full_result.append(rmT)

            # 전체 응답을 합치고
            full_message = "\n\n".join(full_result)

            # Slack 메시지 추출 및 전송
            slack_message = extract_slack_message(full_message)
            send_to_slack(slack_message)

            # Streamlit용 출력은 Slack 메시지를 제외한 나머지만 출력
            streamlit_only_output = full_message.split("*✅ 코드 룰셋 검사 결과*")[0].strip()
            if streamlit_only_output:
                col2.markdown(streamlit_only_output)

        if 'previous_question' not in st.session_state:
            st.session_state.previous_question = ""