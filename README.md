# 🚀 PDEA College Assistant Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG) chatbot** built for **Pune District Education Association (PDEA)** colleges to provide accurate, real-time responses for student queries like admissions, courses, fees, and placements.

> ⚡ **Production-ready AI chatbot (Under approval by PDEA)**

---

## 📌 Project Overview

The **PDEA College Assistant** uses **RAG architecture** to deliver context-aware and reliable answers by retrieving relevant data from a custom knowledge base.

🎯 **Goal:**  
Automate student queries and reduce manual workload using AI.

---

## ✨ Features

- 🤖 AI-powered chatbot  
- 📚 Handles queries:
  - Admissions  
  - Courses (Arts, Commerce, Science)  
  - Fees  
  - Placements  
  - Hostel & Facilities  
- 🔍 RAG-based retrieval (reduces hallucination)  
- 🧠 Context-aware responses  
- 💬 Streamlit UI  
- ⚡ Fast & scalable  

---

## 🧠 Tech Stack

| Category        | Technology |
|----------------|-----------|
| Language       | Python |
| Framework      | Streamlit |
| LLM            | OpenAI API |
| Vector DB      | FAISS |
| Embeddings     | OpenAI / HuggingFace |
| NLP Framework  | LangChain |
| Deployment     | Local (Cloud Ready) |

---

## 🏗️ System Architecture

```
User Query → Embedding → FAISS Vector DB
                     ↓
           Relevant Documents Retrieved
                     ↓
                LLM (RAG)
                     ↓
           Context-Aware Response
                     ↓
                Streamlit UI
```

---

## 📂 Project Structure

```
PDEA-Chatbot/
│
├── PDEA Chatbot/
│   ├── Collections/
│   │   ├── chatbot_ui.png
│   │   ├── College_Chatbot_Training_Questions.pdf
│   │   └── PDEA_Chatbot_Proposal.pdf
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── data/
│   │   ├── embeddings.npy
│   │   ├── faiss.index
│   │   ├── metadata.json
│   │   └── pdea_chatbot_qa.json
│   │
│   ├── logs/
│   │   └── unknown_questions.txt
│   │
│   ├── scripts/
│   │
│   ├── src/
│   │   ├── api.py
│   │   ├── embedder.py
│   │   ├── gemini_api.py
│   │   ├── main.py
│   │   ├── rag_pipeline.py
│   │   └── vector_store.py
│   │
│   ├── PDEA_Logo.png
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/tanmay302/PDEA_chatbot.git
cd PDEA_chatbot
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add API Key
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 5️⃣ Run the App
```bash
streamlit run app.py
```

---

## 📸 Demo (UI Screenshot)

<p align="center">
  <img src="PDEA Chatbot/Collections/chatbot_ui.png" alt="Banner" height ="30%" width="100%">
</p>

---

## 🔍 Example Queries

- What courses are offered?  
- What are the fees?  
- How to apply for admission?  
- Is hostel available?  
- Tell me about placements  

---

## 🚀 Future Improvements

- 🎤 Voice-based chatbot  
- ☁️ Cloud deployment (AWS/GCP)  
- 📱 Mobile UI  
- 📊 Admin dashboard  
- 📄 Auto document ingestion  

---

## 🏫 About PDEA

Developed for **Pune District Education Association (PDEA)**  
Enhancing education with AI-driven solutions.

---

## 👨‍💻 Author

**Tanmay Khedekar**  
🔗 https://github.com/tanmay302  

---

## 💡 Repository Description (Use on GitHub)

AI-powered RAG chatbot for college enquiry automation using FAISS, LangChain, and LLMs (Under approval by PDEA)
