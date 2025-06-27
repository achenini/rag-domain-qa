from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os


def load_documents(folder_path):
    documents = []
    for fname in os.listdir(folder_path):
        with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
            documents.append(Document(page_content=f.read()))
    return documents

def create_vectorstore(documents):
    # Class that splits each document into smaller chunks with an overlap
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # perform split
    texts = splitter.split_documents(documents)
    # use pre-trained embedding model that turns each text chunk into a vector representation
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # store vectors using a similarity search/clustering algorithm combo for vectors
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def answer_query(query, vectorstore):
    # convert vectorstore to a retriever object with query as the input parameter
    retriever = vectorstore.as_retriever()
    # returns top matching document chunks
    docs = retriever.get_relevant_documents(query)
    # loads q&a chain via Hugging Face Models
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")  # Or any model you have
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)