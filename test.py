
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
import os
os.environ["OPENAI_API_KEY"] = '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'
loader = TextLoader('law.txt',encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
#print(docs)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={
        "uri": "https://in01-5517cef144b426e.ali-cn-beijing.vectordb.zilliz.com.cn:19530",
        "user": "db_admin",
        "password": "Uf1%{F.5G4xo8H^%",
        #"secure": True
    }
)
