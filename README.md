
# 📄 Dokumentasi Menjalankan Chatbot BPJS Kesehatan (Streamlit + LangChain + Ollama)

## 📌 1. Persyaratan

Pastikan sudah menginstal dependency berikut (versi diambil dari hasil cek kamu):

* `langchain==0.3.27`
* `torch==2.6.0`
* `streamlit==1.45.0`
* `faiss==1.12.0`
* `nltk==3.9.1`
* `tqdm==4.67.1`

Tambahan (opsional tapi direkomendasikan):

* `bert_score` → untuk evaluasi model berbasis BERT
* `rouge-score` → untuk evaluasi teks berbasis ROUGE
* `langchain-huggingface`
* `langchain-community`
* `langchain-ollama`

Install semua package dengan:

```bash
pip install langchain==0.3.27 torch==2.6.0 streamlit==1.45.0 faiss-cpu==1.12.0 nltk==3.9.1 tqdm==4.67.1
pip install langchain-huggingface langchain-community langchain-ollama
pip install bert-score rouge-score
```

> Jika pakai GPU, ganti `faiss-cpu` dengan `faiss-gpu`.

---

## 📌 2. Persiapan Dataset

Pastikan file dataset sudah ada di:

```
/Users/muhammadzuamaalamin/Documents/fintunellm/deploy/dataset_qa copy.json
```

Format JSON harus seperti ini:

```json
[
  {
    "question": "Apa itu BPJS Kesehatan?",
    "answer": "BPJS Kesehatan adalah badan penyelenggara jaminan sosial di bidang kesehatan."
  },
  {
    "question": "Bagaimana cara mendaftar BPJS Kesehatan?",
    "answer": "Pendaftaran dapat dilakukan melalui kantor BPJS Kesehatan atau aplikasi Mobile JKN."
  }
]
```

---

## 📌 3. Persiapan Model Embedding

Pastikan model embedding **BGE-M3** sudah ada di folder:

```
/Users/muhammadzuamaalamin/Documents/fintunellm/model/bge-m3
```

Kalau belum, bisa download dari HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-m3 /Users/muhammadzuamaalamin/Documents/fintunellm/model/bge-m3
```

---

## 📌 4. Menjalankan Ollama

Pastikan **Ollama** sudah terinstal dan model `qwen2.5:7b` tersedia.

Cek model:

```bash
ollama list
```

Kalau belum ada, jalankan:

```bash
ollama pull qwen2.5:7b
```

---

## 📌 5. Menjalankan Aplikasi

Pindah ke direktori tempat `app.py`, lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi akan berjalan di browser pada:

👉 [http://localhost:8501](http://localhost:8501)

---

## 📌 6. Fitur Aplikasi

* Chatbot berbasis **RAG (Retrieval-Augmented Generation)**
* Pertanyaan dijawab berdasarkan dataset + model Ollama (`qwen2.5:7b`)
* Riwayat percakapan tersimpan dalam session
* Bisa melihat **konteks dokumen** yang digunakan untuk menjawab

---

## 📌 7. Catatan

* Jika ingin menambahkan dataset baru, cukup update file JSON lalu hapus folder `faiss_index_bpjs` agar index dibuat ulang.
* Performansi chatbot tergantung pada kecepatan embedding model (`bge-m3`) dan LLM Ollama (`qwen2.5:7b`).
* Jalankan `ollama serve` di background jika model tidak merespon.

---
