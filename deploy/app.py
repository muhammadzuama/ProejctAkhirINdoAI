# app.py
from flask import Flask, render_template, request
from langchain.schema import Document
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Inisialisasi Flask
app = Flask(__name__)

# --- 1. Muat dataset
def load_documents():
    with open('/Users/muhammadzuamaalamin/Documents/fintunellm/rag/dataset_qa.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            page_content=f"Pertanyaan: {item['question']}\nJawaban: {item['answer']}",
            metadata={"source": "bpjs_qa_dataset"}
        ) for item in data
    ]

# --- 2. Inisialisasi model embedding
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# --- 3. Inisialisasi FAISS (buat atau muat)
def get_vectorstore():
    index_path = "faiss_index_bpjs"
    embedding_model = get_embedding_model()

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        print("📂 Memuat FAISS index dari disk...")
        vectorstore = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # ✅ Cek apakah dimensi FAISS cocok dengan embedding model
        test_vec = embedding_model.embed_query("cek dimensi")
        if vectorstore.index.d != len(test_vec):
            print(f"⚠️ Dimensi FAISS ({vectorstore.index.d}) "
                  f"≠ dimensi embedding ({len(test_vec)}).")
            print("🔄 Membuat ulang index...")
            docs = load_documents()
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local(index_path)
            print(f"✅ Index FAISS tersimpan ulang di {index_path}")
    else:
        print("🧠 Membuat FAISS index baru...")
        docs = load_documents()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(index_path)
        print(f"✅ Index FAISS tersimpan di {index_path}")
    
    return vectorstore

# --- 4. Inisialisasi LLM dan Prompt
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

# --- Inisialisasi global (hanya sekali saat startup)
print("🚀 Memulai: Memuat model dan data...")
rag_chain, retriever = get_rag_chain()
print("✅ Sistem siap!")

# --- Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form["question"].strip()
        
        try:
            # Retrieve konteks
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate jawaban
            response = rag_chain.invoke({
                "question": question,
                "context": context
            })
            answer = response.content
        except Exception as e:
            answer = f"Maaf, terjadi kesalahan: {str(e)}"
    
    return render_template("index.html", answer=answer)

# --- Jalankan aplikasi
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
