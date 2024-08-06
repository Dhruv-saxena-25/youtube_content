from src.data_converter import load_text, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma 
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

## Storing chunks into the Chromadb 
def vector_index():
    document = load_text()
    text_chunks = text_splitter(document)
    embedding = load_embedding()
    #storing vector in choramdb
    vectordb = Chroma.from_documents(text_chunks, embedding=embedding, persist_directory='./db')
    vectordb.persist()


if __name__ == "__main__":
    vector_index()
    



