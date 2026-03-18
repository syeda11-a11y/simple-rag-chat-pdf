# Chat with PDF – Simple RAG 📄🤖

A minimal Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about its content in a conversational chat interface. It also includes a **Job Search** feature that scans today's live job postings across Google Jobs (and LinkedIn, Indeed, Glassdoor via Google) to find roles that match your uploaded resume.

## Features

- **Streamlit UI** – drag-and-drop PDF upload, a chat interface, and a job-search tab
- **LangChain RAG pipeline** – PDF loading → chunking → embedding → retrieval → generation
- **FAISS vector store** – runs entirely in-memory/locally, no cloud database required
- **OpenAI** – `text-embedding-ada-002` for embeddings and `gpt-3.5-turbo` for the LLM
- **Conversational memory** – follow-up questions are understood in context
- **Job Search tab** – upload your resume and get today's matching job postings from Google Jobs (LinkedIn, Indeed, Glassdoor, and more) powered by [SerpAPI](https://serpapi.com/)

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- A [SerpAPI key](https://serpapi.com/) *(free tier: 100 searches/month – needed only for the Job Search tab)*

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
echo "OPENAI_API_KEY=sk-..." > .env
echo "SERPAPI_API_KEY=your-serpapi-key" >> .env
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

1. Enter your **OpenAI API key** in the sidebar (or set it via `.env`).
2. Enter your **SerpAPI key** in the sidebar if you want to use the Job Search tab.
3. Upload a **PDF file** (your resume works best for the Job Search tab).
4. Use the **💬 Chat with PDF** tab to ask questions about the document.
5. Switch to the **🔍 Find Matching Jobs** tab to search for today's job postings.

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

### Job Search (new)
1. **Resume Parsing** – The uploaded PDF is loaded and its full text is extracted.
2. **Keyword Extraction** – GPT-3.5-turbo identifies the most relevant job titles and skills from the resume text.
3. **Live Job Search** – The extracted keywords are used to query **Google Jobs** via SerpAPI, filtered to postings from **today** (`date_posted:today`). Results include listings from LinkedIn, Indeed, Glassdoor, company career pages, and more.
4. **Results Display** – Each matching job is shown with title, company, location, posting date, schedule type, a short description preview, and a direct apply/view link.
