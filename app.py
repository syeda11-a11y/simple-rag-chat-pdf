import json
import os
import tempfile
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient

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
    """Extract raw text from a resume PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.remove(tmp_path)

    return "\n".join(page.page_content for page in pages)


def extract_job_profile(resume_text: str, openai_api_key: str) -> dict:
    """Use the LLM to extract a structured job profile from the resume text."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    prompt = (
        "You are a career advisor. Read the resume below and respond with a JSON object "
        "containing these keys:\n"
        "  - job_titles: list of 3 relevant job title search terms (e.g. 'Software Engineer', "
        "'Data Scientist')\n"
        "  - skills: list of up to 8 key technical skills or keywords\n"
        "  - experience_level: one of 'entry', 'mid', 'senior', or 'lead'\n\n"
        "Respond ONLY with valid JSON, no extra text.\n\n"
        # Truncate to ~6 000 chars (~1 500 tokens) to stay within gpt-3.5-turbo's context window cost-effectively
        f"Resume:\n{resume_text[:6000]}"
    )
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: return minimal profile so job search still runs even if JSON parsing fails
        return {"job_titles": ["Software Engineer"], "skills": [], "experience_level": "mid"}


def search_jobs_today(profile: dict, tavily_api_key: str) -> list[dict]:
    """Search LinkedIn, Indeed, and other job boards for today's postings matching the profile."""
    client = TavilyClient(api_key=tavily_api_key)
    today = date.today().strftime("%B %d, %Y")  # e.g. "March 18, 2026"

    skills_str = " ".join(profile.get("skills", [])[:5])
    titles = profile.get("job_titles", ["Software Engineer"])

    queries = []
    for title in titles[:2]:
        queries.append(
            f'site:linkedin.com/jobs "{title}" {skills_str} jobs posted today {today}'
        )
        queries.append(
            f'site:indeed.com "{title}" {skills_str} jobs posted today {today}'
        )
    # Broader search across all career sites
    queries.append(
        f'"{titles[0]}" {skills_str} jobs posted today {today} -site:glassdoor.com'
    )

    seen_urls: set[str] = set()
    results: list[dict] = []

    for query in queries:
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=False,
            )
            for item in response.get("results", []):
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(
                        {
                            "title": item.get("title", "Job Posting"),
                            "url": url,
                            "snippet": item.get("content", "")[:300],
                            "source": _source_label(url),
                        }
                    )
        except Exception as exc:
            # Log the failure so users can debug; continue so other queries still run
            st.warning(f"A search query failed and was skipped: {exc}")
            continue

    return results


def _source_label(url: str) -> str:
    """Return a human-readable source label based on the URL's hostname."""
    from urllib.parse import urlparse

    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        hostname = ""

    # Match on the registered domain portion of the hostname (e.g. "linkedin.com" in "www.linkedin.com")
    if hostname == "linkedin.com" or hostname.endswith(".linkedin.com"):
        return "LinkedIn"
    if hostname == "indeed.com" or hostname.endswith(".indeed.com"):
        return "Indeed"
    if hostname == "glassdoor.com" or hostname.endswith(".glassdoor.com"):
        return "Glassdoor"
    if hostname == "monster.com" or hostname.endswith(".monster.com"):
        return "Monster"
    if hostname == "ziprecruiter.com" or hostname.endswith(".ziprecruiter.com"):
        return "ZipRecruiter"
    if hostname in {"wellfound.com", "angel.co"} or hostname.endswith((".wellfound.com", ".angel.co")):
        return "Wellfound"
    if hostname == "dice.com" or hostname.endswith(".dice.com"):
        return "Dice"
    return "Career Site"


def render_job_finder(openai_api_key: str, tavily_api_key: str) -> None:
    """Render the Job Finder tab UI."""
    st.subheader("🔍 Find Jobs Matching Your Resume")
    st.write(
        "Upload your resume (PDF) and click **Find Jobs** to discover positions "
        f"posted today ({date.today().strftime('%B %d, %Y')}) on LinkedIn, Indeed, and more."
    )

    resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf", key="resume_uploader")

    if resume_file:
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            return
        if not tavily_api_key:
            st.info("Please add your Tavily API key in the sidebar to continue.")
            return

        if st.button("🚀 Find Matching Jobs Posted Today"):
            with st.spinner("Reading your resume…"):
                resume_text = extract_resume_text(resume_file.getvalue())

            with st.spinner("Extracting your job profile…"):
                profile = extract_job_profile(resume_text, openai_api_key)

            st.success("Resume analyzed!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Suggested Job Titles**")
                for t in profile.get("job_titles", []):
                    st.write(f"• {t}")
            with col2:
                st.markdown("**Key Skills Detected**")
                for s in profile.get("skills", []):
                    st.write(f"• {s}")

            with st.spinner("Searching job boards for today's postings…"):
                jobs = search_jobs_today(profile, tavily_api_key)

            if jobs:
                st.markdown(f"### 📋 {len(jobs)} Matching Job(s) Found Today")
                for job in jobs:
                    with st.container(border=True):
                        badge = f"🏷️ **{job['source']}**"
                        st.markdown(f"{badge}  \n[**{job['title']}**]({job['url']})")
                        if job["snippet"]:
                            st.caption(job["snippet"])
            else:
                st.warning(
                    "No jobs found for today's date. "
                    "Try again later or broaden your search by updating your resume keywords."
                )


def main():
    st.set_page_config(page_title="Chat with PDF & Job Finder", page_icon="📄")
    st.header("Chat with your PDF (Simple RAG) 🤖")

    # 1. Sidebar for API Keys (or use .env)
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

        st.divider()

        tavily_api_key = st.text_input(
            "Tavily API Key (for Job Search)",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
        )
        st.markdown("[Get a free Tavily API key](https://app.tavily.com/)")

    # 2. Tabs: Chat with PDF | Job Finder
    tab_chat, tab_jobs = st.tabs(["💬 Chat with PDF", "💼 Job Finder"])

    # ── Chat with PDF tab ────────────────────────────────────────────────────
    with tab_chat:
        pdf_file = st.file_uploader("Upload your PDF", type="pdf", key="chat_pdf")

        if pdf_file:
            if not openai_api_key:
                st.info("Please add your OpenAI API key in the sidebar to continue.")
                st.stop()

            pdf_bytes = pdf_file.getvalue()
            qa_chain = build_qa_chain(pdf_bytes, openai_api_key)

            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

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

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append((prompt, answer))

    # ── Job Finder tab ───────────────────────────────────────────────────────
    with tab_jobs:
        render_job_finder(openai_api_key, tavily_api_key)


if __name__ == "__main__":
    main()
