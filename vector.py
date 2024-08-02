import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sentence_transformers
import torch

DATA_PATH = 'Data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# create vector_store

def vector_store():
    loader = DirectoryLoader(DATA_PATH , glob='*.pdf' , loader_cls= PyPDFLoader)
    
    documents = loader.load()
    
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
    text = text_splitters.split_documents(documents)
    
    
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2' , model_kwargs = {'device':'cpu'})
    
    db = FAISS.from_documents(text , embeddings)
    db.save_local(DB_FAISS_PATH)
    

if __name__ == "__main__":
    vector_store()