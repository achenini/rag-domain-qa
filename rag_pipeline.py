from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os

def load_documents(folder_path):
    documents = []
    for fname in os.listdir(folder_path):
        with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
            documents.append(Document(page_content=f.read()))
    return documents

def create_vectorstore(documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def answer_query(query, vectorstore):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    return chain.run(input_documents=docs, question=query)
