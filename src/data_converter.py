from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import YoutubeLoader
from fpdf import FPDF
import os



pdf = FPDF()

## Convert YouTube video URL into PDF file "data/yt_docs.pdf"

def url_ingest(url):
    loader = YoutubeLoader.from_youtube_url(url)
    # url_id = url.split("/")[-1].split("=")[-1]
    text = loader.load()
    save_text= text[0].page_content
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.write(5, save_text)
    os.makedirs("data",  exist_ok= True)
    dir_path= os.path.join("data", "yt_docs.pdf")
    pdf.output(dir_path)
 


## Loading "yt_docs.pdf" file as documents
def load_text():
    loader = PyPDFLoader("data\yt_docs.pdf")
    document = loader.load()
    return document




## Creating documents into text chunks
def text_splitter(document):

    splitter = RecursiveCharacterTextSplitter(chunk_size= 2500, chunk_overlap= 200)
    text_chunks = splitter.split_documents(document)
    return text_chunks


## Loading the google embedding
def load_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings



if __name__ == "__main__":
    url_ingest("https://www.youtube.com/watch?v=BVKGVDQ32Cc&t=2636s")
    document= load_text()
    chunks= text_splitter(document)
    print(len(chunks))