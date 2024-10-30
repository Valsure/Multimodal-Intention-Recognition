import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import ChatGLM
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import jsonlines
import json

def load_documents(directory="/hy-tmp/project/WWW2025/liziheng/documents"):
    loader = DirectoryLoader(directory)
    # loader = UnstructuredFileLoader(directory)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def load_embedding_mode(embedding_model_path):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceBgeEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def store_chroma(docs, embeddings, persist_directory):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return db

def read_jsonl(path): #加载问题
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content



# while True:
#     query = input("请输入：")
#     if query.lower() == "exit":
#         break
#     resp = qa.invoke(query)['result'].replace('\\n', '\n')
#     print(resp)

if __name__ == "__main__":
    
        # 加载嵌入模型
    embeddings = load_embedding_mode("/hy-tmp/project/WWW2025/models/bge-large-zh-v1.5")

    # 检查并更新 VectorStore

    documents = load_documents()
    db = store_chroma(documents, embeddings, "/hy-tmp/project/WWW2025/VectorStore_small")


    # 初始化 LLM
    api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm = ChatOpenAI(model="qwen2-7b-instruct", openai_api_base=api_base, openai_api_key="sk-33c0665bdb79433ea0820a67bb059d95")

    # 重新初始化检索器
    retriever = db.as_retriever()

    # 创建 RetrievalQA 实例
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True
    )
    
    
    query = "小明喜欢吃什么水果"
    resp = qa.invoke(query)['result'].replace('\\n', '\n')
    print(resp)
    