import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2.errors import PdfReadError

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files, handling errors gracefully."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except PdfReadError:
            st.warning(f"Unable to read the file: {pdf.name}. Skipping...")
        except Exception as e:
            st.warning(f"An error occurred while processing the file: {pdf.name}. Skipping...")
    return text

def get_txt_text(txt_docs):
    """Extract text from uploaded TXT files, handling decoding errors gracefully."""
    text = ""
    for txt_file in txt_docs:
        try:
            file_content = txt_file.read().decode("utf-8")
            text += file_content
        except UnicodeDecodeError:
            st.warning(f"Unable to decode the file: {txt_file.name}. It may not be in UTF-8 format.")
        except Exception as e:
            st.warning(f"An unexpected error occurred while reading the file: {txt_file.name}. Skipping...")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save a FAISS index from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the conversational chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context," without guessing.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def main():
    st.set_page_config("Chatbot for Your Documents")
    st.header("Chatbot for Your Documents")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file uploads
    with st.sidebar:
        st.title("Upload Files")
        uploaded_files = st.file_uploader(
            "Select your Pdf or Text files.",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )
        if uploaded_files:
            # Show progress only for file uploads
            with st.spinner("Processing files..."):
                text = ""
                for file in uploaded_files:
                    if file.name.endswith(".pdf"):
                        text += get_pdf_text([file])
                    elif file.name.endswith(".txt"):
                        text += get_txt_text([file])
                if text.strip():
                    text_chunks = get_text_chunks(text)
                    get_vector_store(text_chunks)
                    st.success("Upload successful.")
                else:
                    st.warning("Uploaded files did not contain any extractable text. Please try again.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input and processing
    user_question = st.chat_input("Ask a question")
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_question)

        # Process the user's question (no spinner for this part)
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            index_path = "faiss_index"
            if not os.path.exists(os.path.join(index_path, "index.faiss")):
                raise FileNotFoundError(f"FAISS index file not found at {index_path}/index.faiss. Please process the files first.")
            new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            assistant_response = response["output_text"]
        except FileNotFoundError as e:
            assistant_response = f"Error: {e}"
        except Exception as e:
            assistant_response = f"An unexpected error occurred: {e}"

        # Display assistant's response in chat
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
