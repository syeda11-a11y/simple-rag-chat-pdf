# Chat with PDF – Simple RAG 📄🤖

A minimal Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about its content in a conversational chat interface.

## Features

- **Streamlit UI** – drag-and-drop PDF upload and a chat interface
- **LangChain RAG pipeline** – PDF loading → chunking → embedding → retrieval → generation
- **FAISS vector store** – runs entirely in-memory/locally, no cloud database required
- **OpenAI** – `text-embedding-ada-002` for embeddings and `gpt-3.5-turbo` for the LLM
- **Conversational memory** – follow-up questions are understood in context

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/syeda11-a11y/simple-rag-chat-pdf.git
cd simple-rag-chat-pdf

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your API key to a .env file so you don't have to type it every time
echo "OPENAI_API_KEY=sk-..." > .env
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

1. Enter your OpenAI API key in the sidebar (or set it via `.env`).
2. Upload a PDF file.
3. Ask questions in the chat box – the model will answer based only on the PDF content.

## Project Structure

```
simple-rag-chat-pdf/
├── app.py            # Complete application logic and Streamlit UI
├── requirements.txt  # Python dependencies
├── .gitignore        # Standard Python gitignore
└── README.md         # This file
```

## How It Works

1. **PDF Loading** – `PyPDFLoader` extracts text page-by-page.
2. **Chunking** – `RecursiveCharacterTextSplitter` splits the text into overlapping chunks so context is not lost at boundaries.
3. **Embedding & Indexing** – `OpenAIEmbeddings` converts each chunk to a vector, stored in a local `FAISS` index.
4. **Retrieval** – The top-k most similar chunks are fetched for each user question.
5. **Generation** – `ChatOpenAI` (GPT-3.5-turbo) answers the question using the retrieved chunks and the conversation history via `ConversationalRetrievalChain`.
