from langchain.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA 
from src.data_converter import load_embedding
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
    Use your own knowledge to enhance the answer on the basis of context. also if required 
    generate some images to explain the things
    you are also content write for the youtube video. Ensure your answers are relevant to the 
    question context. Your responses should be concise and informative.
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


# if __name__=='__main__':
    # url_ingest("https://www.youtube.com/watch?v=BVKGVDQ32Cc&t=2636s")
    # document= load_text()
    # chunks= text_splitter(document)
    # vector_index()
    # chain  = generation()
    # result= chain("can you tell me the summary of video in 50 lines?")
    # print(result['result'])


