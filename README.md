# UDST Policy Chatbot 📚🤖

## Overview
The **UDST Policy Chatbot** is an AI-powered assistant designed to provide quick and accurate answers to questions about **University of Doha for Science & Technology (UDST)** policies. It utilizes **retrieval-augmented generation (RAG)** to fetch relevant policy information and present structured responses.

## Features
- 📜 **Fetch and process university policies** from official UDST web pages.
- 🤖 **AI-powered chatbot** for answering policy-related questions.
- 🔍 **Retrieval-Augmented Generation (RAG)** to enhance accuracy.
- 📄 **Automatic PDF generation** of policy documents.
- 🌐 **Streamlit Web Interface** for an interactive chatbot experience.

---

## Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/NajlaZuhir/Agnetic_UDST_Chatbot.git
cd Agnetic_UDST_Chatbot
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=your_openai_api_key
```

---

## Usage
### Run the Chatbot
```bash
streamlit run app.py
```

### Ask a Policy Question
- Example: *"How many absences are allowed before I risk failing?"*
- The chatbot will return relevant policy details along with official UDST references.

---

## Project Structure
```
📂 udst-policy-chatbot
│-- 📜 Document.py (Extracts and saves policy data as PDFs)
│-- 🤖 agnetic_rag_policies.py (AI-powered policy retrieval & RAG processing)
│-- 🌐 app.py (Streamlit web app for chatbot UI)
│-- 📄 requirements.txt (Dependencies list)
│-- 🔑 .env (Stores OpenAI API key - NOT included in repo)
│-- 📁 policy_pdfs/ (Stores downloaded policy documents)
```

---

## Technologies Used
- 📝 **BeautifulSoup & Requests** - Web scraping for policy extraction
- 📄 **FPDF** - PDF generation for offline policy storage
- 🤖 **LlamaIndex** - RAG implementation for policy retrieval
- 🔥 **OpenAI GPT** - AI-driven chatbot responses
- 🌐 **Streamlit** - Web UI for chatbot interaction

---

## Contributing
Feel free to fork this repository, submit issues, or suggest improvements! 🚀



