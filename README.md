# 🎧 Enterprise Customer Support System

> AI-powered customer support built with LangChain, GROQ, and Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-support-wwa8prrtl5opqpsoqzwdwm.streamlit.app/)

---

## 🌐 Live Demo

**[👉 Open App](https://customer-support-wwa8prrtl5opqpsoqzwdwm.streamlit.app/)**

---

## 📌 Overview

An intelligent enterprise customer support system that uses **Retrieval-Augmented Generation (RAG)** to answer customer questions across three product knowledge bases. The AI agent can troubleshoot issues, check ticket status, calculate plan upgrades, and create new support tickets — all in real time.

---

## ✨ Features

- 🤖 **AI Support Agent** — Powered by `llama-3.3-70b-versatile` via GROQ API
- 📚 **3 Product Knowledge Bases** — CloudStore Pro, SecureVault AI, DataFlow Analytics
- 🔍 **TF-IDF RAG Pipeline** — Lightweight retrieval without PyTorch or HuggingFace
- 🧠 **Conversation Memory** — Remembers context across the full session
- 🎯 **Product Filter** — Focus the agent on a specific product
- 🎫 **Ticket Management** — Check status, create new tickets with SLA tracking
- 💰 **Upgrade Calculator** — Compare plan pricing and benefits
- ⚡ **Quick Questions** — One-click common support scenarios

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | GROQ API — `llama-3.3-70b-versatile` |
| Framework | LangChain + LangGraph |
| Frontend | Streamlit |
| Retrieval | TF-IDF (scikit-learn) |
| Agent Type | ReAct Agent (LangGraph) |
| Memory | Session-based chat history |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-support.git
cd customer-support
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a GROQ API Key
Sign up for free at [console.groq.com](https://console.groq.com) and create an API key.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

Enter your GROQ API Key in the sidebar and start chatting!

---

## 📦 Requirements

```
streamlit>=1.32.0
langchain-groq>=0.1.6
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-text-splitters>=0.2.0
langgraph>=0.1.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

## 🧪 Example Usage

| Question | What the Agent Does |
|----------|-------------------|
| `My CloudStore sync stopped working` | Retrieves troubleshooting steps from knowledge base |
| `Check status of ticket TKT-002` | Calls `check_ticket_status` tool |
| `Upgrade options for CloudStore Pro?` | Calls `calculate_plan_upgrade` tool |
| `I got a breach alert on SecureVault` | Returns security steps from knowledge base |
| `Create a ticket for my dashboard issue` | Calls `generate_support_ticket` tool |

---

## 🏗️ Project Structure

```
customer-support/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📐 Architecture

```
User Input
    │
    ▼
TF-IDF Retriever ──► Top 4 relevant chunks from knowledge base
    │
    ▼
ReAct Agent (llama-3.3-70b-versatile)
    │
    ├── check_ticket_status()
    ├── calculate_plan_upgrade()
    └── generate_support_ticket()
    │
    ▼
Response + Sources displayed in Streamlit UI
```

---

## 📝 Assignment

This project was developed as **Project 7** of the Advanced LangChain Applications assignment, demonstrating:

- ✅ Document vectorization with chunking strategies
- ✅ Minimum 3 distinct product knowledge bases
- ✅ Conversation memory implementation
- ✅ At least 2 specialized support tools (implemented 3)
- ✅ Structured response formatting
- ✅ Error handling and edge case management
- ✅ Streamlit deployment

---

## 👨‍💻 Author

**Youssef Khaled Ismail**

Built with ❤️ using LangChain · GROQ · Streamlit
