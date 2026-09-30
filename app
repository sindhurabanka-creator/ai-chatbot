import streamlit as st
import ollama
import chromdb
from sentence_transformers import SentenceTransformer

sr.set_page_config(page_tilte="Mini RAG Q&A",
page_icon=)

st.title("Mini RAG Q&A")
st.write("Paste a document,store it in ChromaDB,and ask question about it.")

@st.cache_resource
def load_embidding_model():
    return SentenceTranformer("all-MiniLM-L6-v2")

embidding_model=load_embidding_model()

client=chromadb.Client
collection=client.get_or_create_collection(name="document")

document=st.text_area(
    "Paste your document here",
    height=250,
    placeholder="Paste your notes,articles,syllabus,etc"
)

if st.button("Add Document"):
    if not document.strip():
        st.warning("please enter some text.")
    else:
        chunks =[
            document[i:i+500]
            for i in range(0,len(document),500)
        ]

        embeddings=embedding_model.encode(chunks)

        collection.add(
            ids=[f"chunk_(i)" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embbeddings.tolist()
        )

        st.sucess(f"Added{len(chunks0)}  chunk(s)
        to ChromaDB.")
question=st.text_input("? Ask a question about your document")

if st.button("Ask AI"):
    if not question.strip():
        st.warning("please enter a question.")
    elif collection.cont() == 0:
        st.warning("please add a document first.")
    else:
        question_embedding=embedding_model.encode
        ([question])[0]

        results=collection.query(
            query_embedding=[question_embedding.tolist()],
            n_result=min(3,collection.count())
            
        )
        retrived_chunks=results["documents"]["0"]
        context="\n\n".join(retrived_chunks)

        prompt=f"""
You are a helpful AI assitant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

if the answer is not present in the context,
say "I don't know baseda on provided document."
"""
        try:
            response=ollama.chat(
                model="llama3.2",
                message=[{"role":"user","content":prompt}]
            )

            st.subheader("Answer")
            st.write(response["message"]["content"])


            with st.expander("Retrived Context"):
                for i, chunk in enumerator(retrived_chunks):
                    st.write(f"**Chunk{i+1}:**")
                    st.write(chunk)
        except Exception as e:
            st.error(
                "Could not connect to ollama. Make sure ollama is running" " and run'ollama pull llama3.2' first"
            )st.code(str(e))

st.divider()
st.caption("Python+Sentence Transformers+ChromaDB+Ollama+Streamlit")                        
        
        



