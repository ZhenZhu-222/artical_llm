
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

def qa_agent(api, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api)

    # 如果传入的是文件路径，需要打开文件读取内容
    #with open(uploaded_file, "rb") as file:
    content = uploaded_file.read()  # 读取文件内容

    # 将文件内容写入临时 PDF 文件
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(content)

    # 加载 PDF 文件
    loader = PyPDFLoader(temp_pdf_path)
    doc = loader.load()  # 得到 documents 列表

    # 分割文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=40,
        separators=["\n", "\n\n", "。"]
    )
    texts = splitter.split_documents(doc)

    # 生成向量数据库
    embedding_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever()

    # 构建 ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=memory,
        retriever=retriever,
        chain_type="refine"
    )

    # 执行查询
    result = chain.invoke({"chat_history": memory, "question": question})

    return result


# if __name__ == '__main__':
#     memory = ConversationBufferMemory(
#         return_message=True,
#         memory_key="chat_history",
#         output_key="answer"
#     )
#
#     # 本地 PDF 文件路径
#     file = "C:/Users/ROG/Desktop/Code/licode/0 资料/11 项目4：智能PDF问答工具/temp.pdf"
#     question = "今天的内容是什么？"
#
#     response = qa_agent(
#         api="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",
#         memory=memory,
#         uploaded_file=file,  # 传递文件路径
#         question=question
#     )
#
#     print(response["answer"])
