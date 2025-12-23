__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Ericsson Cognitive Assistant", page_icon="📶", layout="wide")
st.title("📶 Ericsson Multi-Modal RAG")
st.markdown("Ask technical questions about the **Ericsson Mobility Report**.")

#api handling
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found in Secrets or Environment.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
PAGES_DIR = os.path.join(BASE_DIR, "data", "processed", "pages")


#the brain
@st.cache_resource
def load_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    template = """
        You are an expert analyst summarizing the Ericsson Mobility Report.
        Use the provided context to answer the user's question.

        Guidelines:
        - If the exact number isn't there, look for trends or estimates in the text.
        - If the context mentions the topic but lacks details, summarize what IS mentioned.
        - Do not invent numbers. If the info is truly missing, just say "The report doesn't specify this detail, but it does mention..."
        - Always reference the page number if available in the context.

        Context:
        {context}

        Question:
        {question}
        """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain, retriever


try:
    chain, retriever = load_chain()
except Exception as e:
    st.error(f"❌ Database Error: {e}")
    st.stop()


def robust_invoke(chain, prompt_text, retries=3):
    for attempt in range(retries):
        try:
            return chain.invoke(prompt_text)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                wait_time = 5
                st.warning(f"API Busy (Attempt {attempt + 1}/{retries}). Retrying...")
                time.sleep(wait_time)
            else:
                return f"Error: {e}"
    return "**API Quota Hit:** I successfully retrieved the data below, but cannot summarize it right now."


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ex: What is the trend for 5G subscriptions?"):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing PDF and Graphs..."):
            docs = retriever.invoke(prompt_text)
            answer = robust_invoke(chain, prompt_text)
            st.markdown(answer)

            failure_triggers = ["i could not find", "does not mention", "no information provided"]
            is_failure = any(trigger in answer.lower() for trigger in failure_triggers)

            if not is_failure:
                with st.expander("Evidence (Text & Graphs)", expanded=True):
                    for doc in docs:
                        raw_source = doc.metadata.get("source", "Unknown")

                        # --- FIX 2: THE WINDOWS PATH CLEANER ---
                        # 1. Normalize slashes (Turn Windows \ into Linux /)
                        clean_source = raw_source.replace("\\", "/")
                        # 2. Split by / and take the last part (the filename)
                        filename = clean_source.split("/")[-1]

                        # 3. Construct the clean Cloud Path
                        cloud_path = os.path.join(PAGES_DIR, filename)

                        page_num = doc.metadata.get("page", "?")
                        doc_type = doc.metadata.get("type", "text")

                        # Display clean filename only
                        st.markdown(f"**Source:** `{filename}` (Page {page_num})")

                        if doc_type == "image":
                            if os.path.exists(cloud_path):
                                st.image(cloud_path, caption=f"Figure from Page {page_num}", width=400)
                            else:
                                # This warning will now show the CLEAN filename, not the full path
                                st.warning(f"Image found in DB but file missing at: {filename}")
                        else:
                            st.info(f"\"{doc.page_content[:300]}...\"")
                        st.divider()
            else:
                st.caption("No relevant evidence chunks passed the relevance filter.")

    st.session_state.messages.append({"role": "assistant", "content": answer})