# 填写您自己的APIKey
import os
os.environ[ 'HF_ENDPOINT'] = 'https://hf-mirror.com'

from myzhipu import ChatZhipuAI
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
ZHIPUAI_API_KEY = "6e137ddcc67cc25f4324e6465e4b5b67.JJvKdTYTdjor4YGG"
llm = ChatZhipuAI(
    temperature=0.1,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)

from pprint import pprint


from langchain_community.vectorstores import Milvus

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader('Law-Book',glob="**/*.md")
data = loader.load()
pprint(data)


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)
pprint(len(all_splits))


from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bgeEmbeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


connection_args = {"host": "127.0.0.1", "port": "19530", "user": "root", "password": "123456"}
# 创建Collection
vector_store = (Milvus(
    embedding_function=bgeEmbeddings,
    connection_args=connection_args,
    collection_name="laws",
    drop_old=True
)).from_documents(
    all_splits,
    embedding=bgeEmbeddings,
    collection_name="laws",
    connection_args=connection_args
)

query = "国旗是？"
docs = vector_store.similarity_search(query)
print(docs)


# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#
# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:
#
# <context>
# {context}
# </context>
#
# 问题: {question}""")
#
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# retriever_chain = (
#     {"context": retriever , "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# pprint(retriever_chain.invoke("抢劫罪怎么处罚？"))

