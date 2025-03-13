import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

model = "gpt-4o-mini"

text_loader_kwargs = {'autodetect_encoding' : True}

folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder , glob = "**/*.md", loader_kwargs=text_loader_kwargs, loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents = chunks , embedding = embeddings)

llm = ChatOpenAI(temperature = 0.7 , model_name = model)

memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm ,memory = memory ,retriever=retriever )

def handle_greeting(user_input):
    import re
    
    cleaned = re.sub(r'[^\w\s]', '', user_input.lower()).strip()
    tokens = cleaned.split()
    
    greeting_words = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'}
    request_keywords = {'menu', 'order', 'get', 'want', 'help', '?', 'what'}
    
    if not tokens:
        return None
    
    if tokens[0] in greeting_words:
        if len(tokens) == 1 or not any(word in request_keywords for word in tokens[1:]):
            return "Welcome to FreshBite Bistro! ☕️ How can I help you today?"
    
    return None

def handle_user_input(user_text):
    greeting_response = handle_greeting(user_text)
    if greeting_response:
        return greeting_response
    
    # Proceed with normal processing if not a greeting
    result = conversation_chain.invoke({'question': user_text})
    return result["answer"]

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Example function that returns a simple response.
# Replace with your real RAG or LangChain logic.


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    user_text = request.values.get('Body', '')
    bot_answer = handle_user_input(user_text)
    twilio_resp = MessagingResponse()
    twilio_resp.message(bot_answer)
    return str(twilio_resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)