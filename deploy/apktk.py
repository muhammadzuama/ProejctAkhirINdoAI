import sys, json, os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton,
    QMessageBox, QLineEdit, QHBoxLayout, QDialog, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

HISTORY_FILE = "chat_history.json"

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

# --- 2. Model embedding
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="/Users/muhammadzuamaalamin/Documents/fintunellm/model/bge-m3")

# --- 3. Vectorstore FAISS
def get_vectorstore():
    index_path = "faiss_index_bpjs"
    embedding_model = get_embedding_model()

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        docs = load_documents()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(index_path)
    return vectorstore

# --- 4. LLM + Prompt
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

# --- 5. History helper
def save_history(question, answer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except:
                history = []
    history.append({"question": question, "answer": answer})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# --- 6. Dialog History
class HistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📜 Riwayat Percakapan")
        self.setGeometry(300, 200, 600, 400)
        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        history = load_history()
        for item in history:
            q = item.get("question", "")
            a = item.get("answer", "")
            lw_item = QListWidgetItem(f"Q: {q}\nA: {a}\n{'-'*50}")
            self.list_widget.addItem(lw_item)

        layout.addWidget(self.list_widget)
        self.setLayout(layout)

# --- 7. PyQt5 GUI
class RagApp(QWidget):
    def __init__(self, rag_chain, retriever):
        super().__init__()
        self.rag_chain = rag_chain
        self.retriever = retriever

        self.setWindowTitle("Asisten BPJS Kesehatan - RAG")
        self.setGeometry(200, 200, 700, 500)

        layout = QVBoxLayout()

        # Title
        title = QLabel("🤖 Asisten BPJS Kesehatan")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Input
        self.label = QLabel("Masukkan Pertanyaan:")
        layout.addWidget(self.label)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Tulis pertanyaan Anda di sini...")
        layout.addWidget(self.input)

        # Tombol
        btn_layout = QHBoxLayout()
        self.button = QPushButton("Cari Jawaban")
        self.button.clicked.connect(self.on_submit)
        btn_layout.addWidget(self.button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)
        btn_layout.addWidget(self.clear_button)

        self.history_button = QPushButton("Lihat History")
        self.history_button.clicked.connect(self.show_history)
        btn_layout.addWidget(self.history_button)

        layout.addLayout(btn_layout)

        # Output
        self.output_label = QLabel("Jawaban:")
        layout.addWidget(self.output_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("background-color: #f5f5f5; padding: 8px;")
        layout.addWidget(self.output)

        self.setLayout(layout)

        # Styling
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 13px;
            }
            QPushButton {
                padding: 6px;
                border-radius: 6px;
                background-color: #1976d2;
                color: white;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

    def on_submit(self):
        question = self.input.text().strip()
        if not question:
            QMessageBox.warning(self, "Error", "Pertanyaan tidak boleh kosong!")
            return

        # tampilkan loading
        self.output.setPlainText("⏳ Sedang memproses jawaban...")

        QTimer.singleShot(100, lambda: self.process_question(question))

    def process_question(self, question):
        try:
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = self.rag_chain.invoke({
                "question": question,
                "context": context
            })
            answer = response.content
            self.output.setPlainText(answer)

            # simpan ke history
            save_history(question, answer)

        except Exception as e:
            self.output.setPlainText(f"❌ Terjadi kesalahan: {str(e)}")

    def clear_text(self):
        self.input.clear()
        self.output.clear()

    def show_history(self):
        dlg = HistoryDialog(self)
        dlg.exec_()

if __name__ == "__main__":
    print("🚀 Memuat model dan index...")
    rag_chain, retriever = get_rag_chain()
    print("✅ Sistem siap!")

    app = QApplication(sys.argv)
    window = RagApp(rag_chain, retriever)
    window.show()
    sys.exit(app.exec_())
