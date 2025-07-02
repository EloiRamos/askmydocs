import os, pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "askmydocs_index.faiss"
META_FILE  = "askmydocs_store.pkl"

embed = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def load_index():
    index = FAISS.load_local(INDEX_FILE, embed, allow_dangerous_deserialization=True)
    # reload metadata
    with open(META_FILE, "rb") as f:
        index.docstore = pickle.load(f)
    return index

def ask_question(query: str) -> str:
    index     = load_index()
    retriever = index.as_retriever()

    llm       = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    qa_chain  = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain({"query": query})["result"]
