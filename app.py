import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("Chat with your PDF (Simple RAG) 🤖")

    # 1. Sidebar for API Key (or use .env)
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

    # 2. File Uploader
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file:
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        # 3. Build (or retrieve cached) QA chain
        pdf_bytes = pdf_file.getvalue()
        qa_chain = build_qa_chain(pdf_bytes, openai_api_key)

        # 4. Chat Interface
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


if __name__ == "__main__":
    main()
