import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from serpapi import GoogleSearch  # type: ignore[import]
    _SERPAPI_AVAILABLE = True
except ImportError:
    _SERPAPI_AVAILABLE = False

# Maximum characters of resume text sent to the LLM for keyword extraction.
# GPT-3.5-turbo has a ~16k-token context window; 4000 chars ≈ ~1000 tokens,
# which is more than enough to capture all key information from a typical resume.
_RESUME_TEXT_LIMIT = 4000

# Load environment variables (API Keys)
load_dotenv()


@st.cache_resource(show_spinner="Processing PDF…")
def build_qa_chain(pdf_bytes: bytes, openai_api_key: str) -> ConversationalRetrievalChain:
    """Build and cache the QA chain for the given PDF content and API key."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        data = loader.load()
    finally:
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )


def extract_resume_text(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF resume."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.remove(tmp_path)

    return "\n".join(page.page_content for page in pages)


def extract_job_keywords(resume_text: str, openai_api_key: str) -> dict:
    """Use OpenAI to extract job titles and key skills from the resume."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)
    prompt = (
        "You are a resume parser. Given the resume text below, extract:\n"
        "1. A short list of the most relevant job titles this person is targeting or has held (max 3).\n"
        "2. A short list of the top technical and professional skills (max 10).\n\n"
        "Respond strictly in this format (no extra text):\n"
        "JOB_TITLES: <comma-separated list>\n"
        "SKILLS: <comma-separated list>\n\n"
        f"Resume:\n{resume_text[:_RESUME_TEXT_LIMIT]}"
    )
    response = llm.invoke(prompt)
    content = response.content

    job_titles: list[str] = []
    skills: list[str] = []
    for line in content.splitlines():
        if line.startswith("JOB_TITLES:"):
            job_titles = [t.strip() for t in line.replace("JOB_TITLES:", "").split(",") if t.strip()]
        elif line.startswith("SKILLS:"):
            skills = [s.strip() for s in line.replace("SKILLS:", "").split(",") if s.strip()]

    return {"job_titles": job_titles, "skills": skills}


@st.cache_data(show_spinner="Searching job postings…", ttl=300)
def search_jobs_today(query: str, serpapi_key: str) -> list[dict]:
    """Search Google Jobs for postings from the last 24 hours via SerpAPI."""
    if not _SERPAPI_AVAILABLE:
        st.error("SerpAPI package not installed. Run: pip install google-search-results")
        return []

    params = {
        "engine": "google_jobs",
        "q": query,
        "chips": "date_posted:today",
        "api_key": serpapi_key,
        "num": "10",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("jobs_results", [])


def render_job_card(job: dict) -> None:
    """Render a single job card in Streamlit."""
    title = job.get("title", "Unknown Title")
    company = job.get("company_name", "Unknown Company")
    location = job.get("location", "")
    via = job.get("via", "")
    description = job.get("description", "")
    detected_extensions = job.get("detected_extensions", {})
    posted = detected_extensions.get("posted_at", "")
    schedule = detected_extensions.get("schedule_type", "")

    # Build an apply link from the job's share link list if available
    apply_url = ""
    for link_info in job.get("apply_options", []):
        apply_url = link_info.get("link", "")
        break

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{title}**")
            st.markdown(f"🏢 {company} &nbsp;|&nbsp; 📍 {location}")
            if via:
                st.markdown(f"_via {via}_")
        with col2:
            if posted:
                st.markdown(f"🕐 {posted}")
            if schedule:
                st.markdown(f"⏱ {schedule}")

        if description:
            with st.expander("Job description"):
                st.write(description[:800] + ("…" if len(description) > 800 else ""))

        if apply_url:
            st.markdown(f"[Apply / View Posting]({apply_url})")

        st.divider()


def job_search_tab(pdf_bytes: bytes, openai_api_key: str, serpapi_key: str) -> None:
    """Render the job-search feature."""
    st.subheader("🔍 Find Today's Matching Job Postings")

    if not serpapi_key:
        st.info(
            "Enter your **SerpAPI key** in the sidebar to search live job postings. "
            "Get a free key at [serpapi.com](https://serpapi.com/)."
        )
        return

    with st.spinner("Analyzing your resume…"):
        resume_text = extract_resume_text(pdf_bytes)
        keywords = extract_job_keywords(resume_text, openai_api_key)

    job_titles = keywords.get("job_titles", [])
    skills = keywords.get("skills", [])

    st.markdown("**Extracted from your resume:**")
    if job_titles:
        st.markdown(f"- 🎯 **Target roles:** {', '.join(job_titles)}")
    if skills:
        st.markdown(f"- 🛠 **Key skills:** {', '.join(skills)}")

    # Build a search query from the top job title + a few skills
    primary_title = job_titles[0] if job_titles else "software engineer"
    top_skills = ", ".join(skills[:4]) if skills else ""
    default_query = f"{primary_title} {top_skills}".strip()

    st.markdown("---")
    custom_query = st.text_input(
        "Search query (editable)",
        value=default_query,
        help="Edit this query and press Enter / click Search to refine results.",
    )

    if st.button("Search Jobs Posted Today", type="primary"):
        jobs = search_jobs_today(custom_query, serpapi_key)
        if jobs:
            st.success(f"Found **{len(jobs)}** job(s) posted today matching your profile.")
            for job in jobs:
                render_job_card(job)
        else:
            st.warning(
                "No jobs found for today's date with this query. "
                "Try broadening the search query or check that your SerpAPI key is valid."
            )


def main():
    st.set_page_config(page_title="Chat with PDF & Job Search", page_icon="📄")
    st.header("Chat with your PDF (Simple RAG) 🤖")

    # 1. Sidebar for API Keys (or use .env)
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

        st.markdown("---")
        serpapi_key = st.text_input(
            "SerpAPI Key (for Job Search)",
            value=os.getenv("SERPAPI_API_KEY", ""),
            type="password",
            help="Required only for the Job Search tab. Free tier available at serpapi.com.",
        )
        st.markdown("[Get a SerpAPI key](https://serpapi.com/)")

    # 2. File Uploader
    pdf_file = st.file_uploader("Upload your Resume or any PDF", type="pdf")

    if pdf_file:
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        pdf_bytes = pdf_file.getvalue()

        # 3. Tabs: Chat | Job Search
        chat_tab, jobs_tab = st.tabs(["💬 Chat with PDF", "🔍 Find Matching Jobs"])

        with chat_tab:
            # Build (or retrieve cached) QA chain
            qa_chain = build_qa_chain(pdf_bytes, openai_api_key)

            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display existing chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle new user input
            if prompt := st.chat_input("Ask something about the PDF"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = qa_chain.invoke(
                        {
                            "question": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )
                    answer = response["answer"]
                    st.markdown(answer)

                    # Update session state with the latest exchange
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append((prompt, answer))

        with jobs_tab:
            job_search_tab(pdf_bytes, openai_api_key, serpapi_key)


if __name__ == "__main__":
    main()

