# streamlit_app.py

# 📦 Import necessary modules
import os  # Used to interact with the file system (e.g., check if files or directories exist)
import streamlit as st  # Streamlit is a library to build interactive web apps
from dotenv import load_dotenv  # Loads environment variables from a .env file
from langchain_openai import OpenAI, OpenAIEmbeddings  # OpenAI LLM and Embedding wrapper for LangChain
from langchain.chains import RetrievalQAWithSourcesChain  # A chain that answers questions and provides sources
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large text into manageable chunks
from langchain_community.document_loaders import UnstructuredURLLoader  # Loads raw data from web URLs
from langchain_community.vectorstores import FAISS  # FAISS is a vector store for fast similarity search

# 📂 Load environment variables from the .env file (like your OpenAI API key)
load_dotenv()

# 🎨 Set up the Streamlit web app interface
st.set_page_config(page_title="Document QA App", layout="centered")  # Set browser tab title and layout
st.title("📚 Document Q&A with Sources")  # Main app heading
st.markdown("Ask questions based on the latest news articles.")  # Brief description or instruction

# 🧠 Initialize the OpenAI language model with settings
llm = OpenAI(
    temperature=0.9,  # Higher temperature makes answers more creative or diverse
    max_tokens=500  # Max length of the generated response
)

# 🌐 Load web content from specified URLs
loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    # Tesla AI article
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
    # Tata Motors Punch iCNG launch
])
data = loaders.load()  # This extracts and loads the text content from those URLs into documents

# ✂️ Break down large documents into chunks that are easier for LLMs to understand and index
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Max characters per chunk
    chunk_overlap=200  # Overlap between chunks to preserve context
)
docs = text_splitter.split_documents(data)  # Split all documents into smaller chunks

# ✅ Feedback for user: show how many chunks were loaded
st.success(f"Loaded {len(docs)} document chunks.")

# 🧠 Create text embeddings (numerical representations) using OpenAI
embeddings = OpenAIEmbeddings()

# 🧠 Generate a vector index from the document chunks using FAISS
vectorindex_openai = FAISS.from_documents(docs, embeddings)

# 💾 Save the FAISS vector index locally so it doesn't need to be recomputed every time
faiss_index_path = "faiss_index"  # Folder name to store the index
vectorindex_openai.save_local(faiss_index_path)  # Save the index to the folder

# 🧲 Load the saved FAISS index from disk if it exists
if os.path.exists(faiss_index_path):
    VectorIndex = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required to allow loading pickled data
    )

# 🔗 Build a QA chain that can retrieve relevant chunks from the index and answer the question using LLM
chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,  # The OpenAI model for generating answers
    retriever=VectorIndex.as_retriever()  # The retriever to fetch relevant document chunks
)

# 🧾 Input field for user to enter their question
user_question = st.text_input(
    "Enter your question here:",
    placeholder="e.g., What is the price of Tiago iCNG?"  # Placeholder hint for the input field
)

# ▶️ When the user enters a question and presses Enter...
if user_question:
    with st.spinner("🔍 Searching for the answer..."):  # Show a loading spinner while processing
        try:
            # Run the QA chain with the user's question
            result = chain({"question": user_question}, return_only_outputs=True)

            # ✅ Display the answer from the LLM
            st.subheader("🧠 Answer")
            st.write(result["answer"])

            # 🔗 Display the sources used to answer the question
            st.subheader("🔗 Sources")
            st.write(result["sources"])

        except Exception as e:
            # ⚠️ If there's any error during processing, show it in the app
            st.error(f"Something went wrong: {e}")
