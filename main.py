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
    Collection
)
ZHIPUAI_API_KEY = "2d17fda9a61aae1d6ac837f9ba9eae91.Ocd1xoebxG7AeGl3"
llm = ChatZhipuAI(
    temperature=0.1,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings


fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
connections.connect("default", host="localhost", port="19530")
print(fmt.format("start connecting to Milvus"))
has = utility.has_collection("laws")
print(f"Does collection hello_milvus exist in Milvus: {has}")
print(fmt.format("Start loading"))
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bgeEmbeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

from langchain_community.vectorstores import Milvus
connection_args = {"host": "127.0.0.1", "port": "19530", "user": "root", "password": "123456"}
laws=Milvus(
    embedding_function=bgeEmbeddings,
    connection_args=connection_args,
    collection_name="laws"
)

from pprint import pprint
retriever = laws.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query = "中华人民共和国的国旗是？"
docs =retriever.invoke(query)
pprint(docs)
# print(len(docs))



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:
<context>
{context}
</context>
问题: {question}""")
retriever_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# ans=retriever_chain.invoke("抢劫罪怎么处理？")
# print(ans)

import time

start = time.time()

# 需要计算的代码块

ans=retriever_chain.invoke("盗窃罪怎么处理？")
print(ans)
end = time.time()
print('程序运行时间为: %s Seconds' % (end - start))