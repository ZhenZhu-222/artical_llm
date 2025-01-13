from langchain_openai import OpenAI
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("南航智能装配智能体")

with st.sidebar:
    openai_api_key = st.text_input("请输入api秘钥：", type="password")
    st.markdown("[请联系赵博18851137913]")

    uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")

# 初始化对话记忆
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
if "memorys" not in st.session_state:
    st.session_state["memorys"] = [
        {"role": "human", "content": "你好"},
        {"role": "ai", "content": "我是南航装配智能助手，有什么可以帮您"}
    ]

# 显示对话历史
for content in st.session_state["memorys"]:
    st.chat_message(content["role"]).write(content["content"])

# 用户输入问题
if "input" not in st.session_state:
    st.session_state["input"] = ""  # 初始化输入框状态

st.session_state["input"] = st.chat_input()


if st.session_state["input"]:
    question = st.session_state["input"]

    if not openai_api_key:
        st.info("请输入你的api秘钥")
        st.stop()

    # 将用户输入加入对话历史
    st.session_state["memorys"].append({"role": "human", "content": question})
    st.chat_message("human").write(question)

    # 处理文件上传和问题
    if uploaded_file:
        with st.spinner("AI正在思考中，请稍等..."):

            try:
                # 调用 `qa_agent` 获取答案
                response = qa_agent(openai_api_key= openai_api_key, st.session_state["memory"],
                                    uploaded_file, question)
                # 将 AI 回复加入对话历史
                st.session_state["memorys"].append({"role": "ai", "content": response["answer"]})
                st.chat_message("ai").write(response["answer"])
            except Exception as e:
                st.error(f"AI处理错误：{str(e)}")

    # 清空输入框
    st.session_state["input"] = ""
