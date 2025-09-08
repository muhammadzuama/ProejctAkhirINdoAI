# app.py (versi Streamlit - Chatbot UI)
import streamlit as st
from langchain.schema import Document
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Muat dataset
@st.cache_resource
def load_documents():
    with open('/Users/muhammadzuamaalamin/Documents/fintunellm/deploy/dataset_qa copy.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            page_content=f"Pertanyaan: {item['question']}\nJawaban: {item['answer']}",
            metadata={"source": "bpjs_qa_dataset"}
        ) for item in data
    ]

# --- 2. Inisialisasi model embedding
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# --- 3. Inisialisasi FAISS
@st.cache_resource
def get_vectorstore():
    index_path = "faiss_index_bpjs"
    embedding_model = get_embedding_model()

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        vectorstore = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # cek dimensi
        test_vec = embedding_model.embed_query("cek dimensi")
        if vectorstore.index.d != len(test_vec):
            docs = load_documents()
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local(index_path)
    else:
        docs = load_documents()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(index_path)
    
    return vectorstore

# --- 4. Inisialisasi RAG
@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)

    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten BPJS Kesehatan. Gunakan konteks untuk menjawab pertanyaan.
Jika tidak tahu, katakan "Saya tidak tahu". Jawab dengan maksimal panjang 5 kalimat.

Pertanyaan: {question}
Konteks: {context}

Jawaban:"""
    )

    return prompt | llm, retriever

# --- UI Streamlit Chatbot
st.set_page_config(page_title="Chatbot BPJS Kesehatan", page_icon="💬")

st.title("💬 Chatbot BPJS Kesehatan")

rag_chain, retriever = get_rag_chain()

# Simpan history percakapan di session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history percakapan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat user
if prompt := st.chat_input("Tanyakan seputar BPJS Kesehatan..."):
    # simpan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": prompt})

    # tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(prompt)

    # proses jawaban bot
    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari jawaban..."):
            try:
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])

                response = rag_chain.invoke({
                    "question": prompt,
                    "context": context
                })

                answer = response.content
                st.markdown(answer)

                # simpan jawaban bot
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # bisa buka konteks jika ingin
                with st.expander("🔎 Lihat konteks yang digunakan"):
                    st.write(context)

            except Exception as e:
                error_msg = f"Maaf, terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
# --- Footer
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px;">
    <div style="text-align: center; font-size: 13px; color: gray;">
        © 2025 Tim Genesis — Muhammad Zuama Al Amin & Juan Hendy Irmanto
    </div>
    """,
    unsafe_allow_html=True
)
