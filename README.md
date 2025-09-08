1. **Mode Online (Streamlit, berbasis web UI)**
2. **Mode Offline (PyQt5, aplikasi desktop GUI)**

Jadi kamu bisa pilih jalankan chatbot di browser atau sebagai aplikasi desktop.

---

# 📄 Dokumentasi Menjalankan Chatbot BPJS Kesehatan

## 📌 1. Persyaratan

Pastikan sudah menginstal dependency berikut (versi berdasarkan hasil pengecekan):

* `langchain==0.3.27`
* `torch==2.6.0`
* `streamlit==1.45.0`
* `faiss==1.12.0`
* `nltk==3.9.1`
* `tqdm==4.67.1`
* `pyqt5` (untuk mode offline/desktop)

Tambahan (opsional tapi direkomendasikan):

* `bert_score` → evaluasi model berbasis BERT
* `rouge-score` → evaluasi teks berbasis ROUGE
* `langchain-huggingface`
* `langchain-community`
* `langchain-ollama`

Install semua package dengan:

```bash
pip install langchain==0.3.27 torch==2.6.0 streamlit==1.45.0 faiss-cpu==1.12.0 nltk==3.9.1 tqdm==4.67.1 pyqt5
pip install langchain-huggingface langchain-community langchain-ollama
pip install bert-score rouge-score
```

> Jika pakai GPU, ganti `faiss-cpu` dengan `faiss-gpu`.

---

## 📌 2. Dataset

Pastikan file dataset tersedia, misalnya:

```
dataset_qa.json
```

Format JSON:

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

## 📌 3. Model Embedding

Chatbot ini menggunakan **BGE-M3** sebagai model embedding (sumber: [HuggingFace – BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)).

### 🔹 Opsi 1 – Offline (Wajib Download Model)

Jika ingin chatbot berjalan **sepenuhnya offline** (tanpa internet), kamu harus mengunduh model embedding secara manual:

```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-m3
```

* Ukuran model: **\~1.34 GB**
* Setelah selesai, akan ada folder `bge-m3` berisi file model (`pytorch_model.bin`, `config.json`, dll).
* Di dalam kode, arahkan path ke folder model lokal, misalnya:

```python
embeddings = HuggingFaceEmbeddings(model_name="path/to/bge-m3")
```

> Catatan: pastikan `git-lfs` sudah terpasang agar file model bisa diunduh dengan benar.

---

### 🔹 Opsi 2 – Online (Model Diunduh Otomatis)

Jika dijalankan **online** dengan internet aktif, HuggingFace akan otomatis mengunduh model saat pertama kali dipanggil:

```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
```

* File model akan tersimpan di cache HuggingFace lokal (biasanya di `~/.cache/huggingface/transformers`).
* Tidak perlu clone manual, tapi membutuhkan koneksi internet stabil.

---

👉 Jadi, pilihannya:

* **Offline mode** → download model manual, besar file ± **1.34 GB**, lalu arahkan path ke folder lokal.
* **Online mode** → cukup panggil nama model `"BAAI/bge-m3"`, akan otomatis terunduh saat pertama kali digunakan.


---

## 📌 4. Menjalankan Ollama

Pastikan Ollama sudah terinstal dan model tersedia:

```bash
ollama list
```

Jika belum ada, jalankan:

```bash
ollama pull qwen2.5:7b
```

---

## 📌 5. Mode Online (Streamlit – Web UI)

File: `app.py`

Jalankan dengan:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser:
👉 [http://localhost:8501](http://localhost:8501)

**Fitur mode online:**

* Chatbot berbasis RAG
* Riwayat percakapan di browser
* Bisa buka konteks dokumen yang dipakai

---

## 📌 6. Mode Offline (PyQt5 – Desktop UI)

File: `rag_desktop.py` (isi kode PyQt5 yang sudah kamu tulis)

Jalankan dengan:

```bash
python rag_desktop.py
```

**Fitur mode offline:**

* Aplikasi desktop mandiri
* Input pertanyaan via GUI
* Riwayat percakapan disimpan ke `chat_history.json`
* Bisa melihat riwayat percakapan melalui tombol **"Lihat History"**

---

## 📌 7. Catatan

* Jika mengganti dataset, hapus folder `faiss_index_bpjs` agar index dibuat ulang.
* Mode **online** cocok untuk deployment (misal ke server/VPS/Streamlit Cloud).
* Mode **offline** cocok digunakan tanpa internet, selama model embedding (`bge-m3`) dan model Ollama (`qwen2.5:7b`) sudah tersedia lokal.
* Jalankan `ollama serve` di background jika model Ollama tidak merespon.

---
