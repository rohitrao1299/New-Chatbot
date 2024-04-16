from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = Flask(__name__)

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries as a Indian Army Officer named as Alpha Bot and Wish our user with Jai Hind. You have not to answer user question other than SSB/AFSB/NSB Interview, Indian Armed Forces "),
        ("user", "Question:{question}")
    ]
)

## Flask framework
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # ollama LLAma2 LLm
        llm = Ollama(model="llama2")
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        chat_history = []
        try:
            with open('chat_history.pkl', 'rb') as f:
                chat_history = pickle.load(f)
        except FileNotFoundError:
            chat_history = []

        chat_history.append((input_text, chain.invoke({"question": input_text})))

        # Save chat history to file
        with open('chat_history.pkl', 'wb') as f:
            pickle.dump(chat_history, f)

        user_input = input_text
        assistant_response = chat_history[-1][-1]
        return render_template('index.html', user_input=user_input, assistant_response=assistant_response, chat_history=chat_history)

    return render_template('index.html')    
if __name__ == '__main__':
    app.run(debug=True)