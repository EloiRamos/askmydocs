# backend/utils.py
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- 1. Extract plain text -------------------------------------
def extract_text(file_obj) -> str:
    """
    Returns all text from a PDF or a plain‑text file‑like object.
    """
    if file_obj.type == "application/pdf":
        text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    # assume .txt for anything else
    return file_obj.read().decode("utf-8")


# ---------- 2. Split text into chunks ---------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Yields chunks of `chunk_size` characters with `overlap` chars overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)
