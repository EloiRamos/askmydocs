import pdfplumber, os, pickle
from backend.utils import extract_text, chunk_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
embed = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "askmydocs_index.faiss"
META_FILE  = "askmydocs_store.pkl"

def process_document(file_obj):
    text   = extract_text(file_obj)
    chunks = chunk_text(text)

def process_document(file_obj):
    # 1  Extract plain text
    if file_obj.type == "application/pdf":
        text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        text = file_obj.read().decode("utf-8")

    # 2  Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_text(text)
    docs     = [Document(page_content=chunk) for chunk in chunks]

    # 3  Create / update FAISS index
    if os.path.exists(INDEX_FILE):
        # load existing index and merge
        index = FAISS.load_local(INDEX_FILE, embed, allow_dangerous_deserialization=True)
        index.add_documents(docs)
    else:
        index = FAISS.from_documents(docs, embed)

    # 4  Persist to disk
    index.save_local(INDEX_FILE)
    # save the store’s doc metadata (needed for reload)
    with open(META_FILE, "wb") as f:
        pickle.dump(index.docstore, f)
