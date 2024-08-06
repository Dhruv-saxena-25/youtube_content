from flask import Flask, render_template,  jsonify, request
from src.retrieval_generation import generation
from src.data_converter import load_embedding, url_ingest, load_text, text_splitter
from src.store_index import vector_index
import shutil
import os 
import sys


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
    return ""


@app.route("/get", methods=["GET", "POST"])

def chat():
    chain = generation()
    msg = request.form["msg"]
    input = msg
    if input == "clear":
        if os.path.exists("data"):
            shutil.rmtree("data")
    

    result=chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug= True)