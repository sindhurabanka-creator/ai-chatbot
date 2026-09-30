import streamlit as st
from pypdf import pdfReader
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Mini RAG Q&A", page_icon="📃")

st.title("Mini RAG Q&A")
st.write("Paste a document, store it in ChromaDB, and ask questions about it.")
st.caption("PDF -> Chunks -> Embeddings -> Chromadb -> Retrival -> Ollama")


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# Create ChromaDB client and collection
client = chromadb.PeristenceClient(path="./chroma_db")
collection = client.get_or_create_collection(name="document")

st.sidebar.header("Settings")
ollama_model=st.sidebar.text_input("Ollama model","llama3.2")
chunk_size=st.sidebar.slider("Chunk size",200,1500,100)
top_k=st.sidebar.slider("Chunks to retrive",1,5,3)
st.header("Build document Store")
uploaded_file=st.file_uploder("Upload a text-based PDF",type=["pdf"])
if uploaded_file and st.button("process & store PDF"):
    reader=PdfReader(uploaded_file)
    text=""
    for page_number,page in enumerate(reader.pages,start=1):
        page_text=page.extract_text()
        if page_text:
            text+=page_text +"\n"

    if not text.strip():
        st.error("No readable text was found.Try a text-based PDF.")
        st.stop()

    Chunks=[]
    for i in range(0,len(text),chunk_size):
        chunk=text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    with st.spinner("Generating embeddings...."):
        enbeddings=model.encode(chunks)

    existing=collection.get()
    if existing["ids"]:
        collection.delete(ids=exists["ids"])

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": uploaded_file.name, "chunk": i}
            for i in range(len(chunks))
        ],
    )

    st.success(f"  Stored {len(chunks)} chunks in ChromaDB.")
    with st.expander("Preview stored chunks"):
        for i, chunk in enumerate(chunks[:5]):
            st.write(f"**Chunk {i + 1}:**")
            st.write(chunk)
            st.divider()

st.header("  Ask Questions")

question = st.text_input(
    "Ask a question about your uploaded document"
)

if st.button(" Retrieve & Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    count = collection.count()

    if count == 0:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    with st.spinner("Searching document..."):
        question_embedding = model.encode([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, count)
        )

    retrieved_chunks = results["documents"][0]

    st.subheader("Retrieved chunks")

    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Retrieved chunk {i + 1}"):
            st.write(chunk)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the concept provided below.
If the answer is not present in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

     with st.spinner(f"Generating answer with {ollama_model}..."):
        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response["message"]["content"]

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(
                f"Could not connect to Ollama. Make sure Ollama is running "
                f"and the model '{ollama_model}' is installed.\n\nError: {e}"
            )







