import os
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    docs = []
    if not os.path.exists(DATA_DIR): return docs
    
    try:
        pdf_loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        docs.extend(pdf_loader.load())
    except: pass
    
    try:
        txt_loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
        docs.extend(txt_loader.load())
    except: pass
    
    return docs

def build_index():
    documents = load_documents()
    if not documents: return None
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print(f"✅ Indexed {len(chunks)} chunks")
    return db

def load_db():
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return build_index()

# LOAD DB
DB = load_db()
print("Database ready!")

def answer_question(question):
    if not DB or not question.strip():
        return "Add files to data/ folder and restart.", ""
    
    try:
        # Get BEST match only
        result = DB.similarity_search(question, k=1)[0]
        
        # SHORT ANSWER (exactly what you want)
        text = result.page_content[:120]
        source = result.metadata.get('source', 'Unknown')
        
        q = question.lower()
        if 'leave' in q or 'annual' in q:
            return f"**LEAVE POLICY:**\n{text}", f"📄 {source}"
        elif 'sick' in q:
            return f"**SICK LEAVE:**\n{text}", f"📄 {source}"
        elif 'health' in q or 'benefit' in q:
            return f"**BENEFITS:**\n{text}", f"📄 {source}"
        else:
            return f"**FOUND:**\n{text}", f"📄 {source}"
    except:
        return "No relevant info.", ""

# UI
with gr.Blocks(title="Knowledge Base Agent") as demo:
    gr.Markdown("# 🤖 Knowledge Base Agent")
    gr.Markdown("**Ask SHORT questions about your documents**")
    
    question = gr.Textbox(label="Question", placeholder="What is annual leave policy?")
    answer = gr.Markdown(label="Answer")
    source = gr.Textbox(label="Source", interactive=False)
    
    gr.Button("🔍 Answer").click(answer_question, question, [answer, source])

if __name__ == "__main__":
    demo.launch(share=False)
