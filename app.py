import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, Runnable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import ChatMessage

# Use a pipeline as a high-level helper
from transformers import pipeline
from typing import Optional

load_dotenv()
print('start!')
# session_state 초기화 함수
def init_service() -> None:
    st.session_state.started = True
    st.session_state.store = dict()
    st.session_state.conversation = dict()
    st.session_state.pipe = pipeline("text-classification", model="Copycats/koelectra-base-v3-generalized-sentiment-analysis")
    
# 대화 기록 출력 함수
def print_conversation(session_id) -> None:
    session_conversation = st.session_state.conversation.get(session_id, [])
    for message in session_conversation:
        role = message.role
        content = message.content
        st.chat_message(f"{role}").write(content)
            
        

# 모델 생성
llm = ChatOpenAI(model='gpt-4o-mini')

# 프롬프트
    # messagesplaceholder 추가
prompt = prompt_template = ChatPromptTemplate.from_messages([
    ("system", "너는 항상 냥체를 사용해서 대답해."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}")
])

# 대화를 저장할 메모리 생성
def get_session_history(session_id) -> BaseChatMessageHistory:
    if not session_id in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 체인 생성
chain : Runnable = prompt | llm

# 체인과 메모리를 연결하기
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history",
)

# Streamlit
    # session_id는 직접 입력 받기
with st.sidebar:
    session_id = st.text_input(label="session Id", value="catgpt1")
    
    if st.button("대화 기록 초기화"):
        st.session_state.store[session_id] = []
        st.session_state.converation[session_id] = []
        st.rerun()    
    # session_state 초기화
if not "started" in st.session_state:
    init_service()
    # session_id 별 대화 저장 공간 초기화
if not session_id in st.session_state.conversation:
    st.session_state.conversation[session_id] = []
    # 입력 받기
if st.session_state.started:
    query = st.chat_input("메세지를 입력하라냥")
    print_conversation(session_id)
    print(session_id)
    if query:
    # 사용자의 입력 저장 및 출력
        st.session_state.conversation[session_id].append(ChatMessage(role='human', content=query))
        st.chat_message("human").write(query)
# invoke
        response = chain_with_memory.invoke({"query": query}, config={"configurable" : {"session_id": session_id}})
        
        # 출력 결과 긍부정 분류
        result = st.session_state.pipe(response.content)
        label = "긍정" if result[0]['label'] == "1" else "부정"
        score = result[0]['score']
        response.content += f"이 답변은 {score * 100:.2f}%의 확률로 {label}적 입니다."
    # 출력 결과 저장 및 출력    
        st.session_state.conversation[session_id].append(ChatMessage(role='ai', content=response.content))
        st.chat_message("ai").write(response.content)
        print(result)

        
        
        
        