# Chat with PDF & Job Finder – Simple RAG 📄🤖💼

A Retrieval-Augmented Generation (RAG) application that lets you:
- **Chat with any PDF** – upload a document and ask questions about its content
- **Find jobs matching your resume** – upload your resume and discover today's job postings on LinkedIn, Indeed, and other career sites

## Features

- **Streamlit UI** – drag-and-drop PDF upload, a chat interface, and a dedicated Job Finder tab
- **LangChain RAG pipeline** – PDF loading → chunking → embedding → retrieval → generation
- **FAISS vector store** – runs entirely in-memory/locally, no cloud database required
- **OpenAI** – `text-embedding-ada-002` for embeddings and `gpt-3.5-turbo` for the LLM
- **Conversational memory** – follow-up questions are understood in context
- **Job Finder** – extracts your skills and job titles from your resume, then searches LinkedIn, Indeed, and more for positions posted today using the [Tavily](https://tavily.com/) search API

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- A [Tavily API key](https://app.tavily.com/) *(free tier available – required for the Job Finder feature)*

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

# 4. (Optional) Add your API keys to a .env file so you don't have to type them every time
echo "OPENAI_API_KEY=sk-..." >> .env
echo "TAVILY_API_KEY=tvly-..." >> .env
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Chat with PDF
1. Enter your OpenAI API key in the sidebar (or set it via `.env`).
2. Select the **💬 Chat with PDF** tab.
3. Upload a PDF file.
4. Ask questions in the chat box – the model will answer based only on the PDF content.

### Job Finder
1. Enter your OpenAI API key **and** your Tavily API key in the sidebar.
2. Select the **💼 Job Finder** tab.
3. Upload your resume as a PDF.
4. Click **🚀 Find Matching Jobs Posted Today**.
5. The app will:
   - Extract your skills, experience level, and target job titles from your resume.
   - Search LinkedIn, Indeed, and other job boards for positions posted **today** that match your profile.
   - Display the results with direct links to apply.

## Project Structure

```
simple-rag-chat-pdf/
├── app.py            # Complete application logic and Streamlit UI
├── requirements.txt  # Python dependencies
├── .gitignore        # Standard Python gitignore
└── README.md         # This file
```

## How It Works

### Chat with PDF
1. **PDF Loading** – `PyPDFLoader` extracts text page-by-page.
2. **Chunking** – `RecursiveCharacterTextSplitter` splits the text into overlapping chunks so context is not lost at boundaries.
3. **Embedding & Indexing** – `OpenAIEmbeddings` converts each chunk to a vector, stored in a local `FAISS` index.
4. **Retrieval** – The top-k most similar chunks are fetched for each user question.
5. **Generation** – `ChatOpenAI` (GPT-3.5-turbo) answers the question using the retrieved chunks and the conversation history via `ConversationalRetrievalChain`.

### Job Finder
1. **Resume Parsing** – `PyPDFLoader` extracts text from your uploaded resume.
2. **Profile Extraction** – GPT-3.5-turbo reads your resume and identifies your target job titles, key skills, and experience level.
3. **Job Search** – Tavily's real-time search API runs targeted queries against LinkedIn, Indeed, and other career sites filtered to today's postings.
4. **Results Display** – Matching jobs are shown with their title, source platform, link, and a short snippet.

