from flask import Flask, render_template,  jsonify, request
from src.data_converter import load_embedding, url_ingest, load_text, text_splitter
from src.store_index import vector_index
import os

from langchain.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA 
from src.data_converter import load_embedding, url_ingest, load_text, text_splitter
from src.store_index import vector_index

from dotenv import load_dotenv
load_dotenv()


def generation():
    embeddings = load_embedding()
    persist_directory = "db"
    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)
    
    retriever= vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8})
    
    YOUTUBE_BOT_TEMPLATE = """
    Your are youtube video url bot is an expert in  giving answers based on the youtube video.
    also if person tells to summarize the video you have to summarize the whole youtube video.
    you can also use your own knowledge to enhance the answer on the basis of context.
    you are also content write for the youtube video. Ensure your answers are relevant to the 
    question context and refrain from straying off-topic. Your responses should be concise and informative.
    Please do not repeat the sentence OR SAME LINES again and again. 
    if someone greets you greet them back.
    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """
    prompt = PromptTemplate(template= YOUTUBE_BOT_TEMPLATE, input_variable= ["context", "question"])

    llm = ChatGoogleGenerativeAI(model= "gemini-1.5-pro", temperature=0.8)

    memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type= "stuff",
                                        retriever= retriever,
                                        input_key= "query",
                                        memory= memory,
                                        chain_type_kwargs= {"prompt": prompt})

    return chain





app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def link():

    if request.method == 'POST':
        user_input = request.form['question']
        url_ingest(user_input)
        document= load_text()
        text_splitter(document)
        vector_index()
        chain = generation()
        # os.system("python src.store_index.py")
    return  str(user_input)


@app.route("/get", methods=["GET", "POST"])

def chat():
    chain = generation()
    msg = request.form["msg"]
    input = msg
    result=chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug= True)