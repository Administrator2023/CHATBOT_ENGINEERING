"""
Master & Client Chatbot for PhD-Level Structural Engineering Exterior Facade Calculations
------------------------------------------------------------------------------------------
This application provides two interfaces:

1. Admin Console (Master):
   - Upload trusted calculation documents (PDF, Excel, or Text) and provide a detailed explanation.
   - Click "Learn" to extract text, combine it with your explanation, split into chunks, generate embeddings via OpenAI,
     and update the knowledge base (stored in a FAISS vector store).
   - The system also queries GPT‑4 (with chain‑of‑thought instructions) to suggest alternative methods and accepts corrections.

2. Client Chatbot:
   - Clients ask structural engineering questions.
   - The system retrieves relevant approved content from the knowledge base and uses GPT‑4 to generate an expert answer.
   - If the retrieval answer is insufficient, it falls back to a full GPT‑4 response.
   
Ensure your OPENAI_API_KEY is set via environment or Streamlit secrets.
"""

import os
import streamlit as st
import pandas as pd
import pdfplumber

# LangChain components for text splitting, vector storage, embeddings, and LLM interaction
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import ChatOpenAI
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Check for API key and initialize session state
# -----------------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OPENAI_API_KEY in your environment or .streamlit/secrets.toml.")
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []  # Holds all approved documents and corrections

# -----------------------------------------------------------------------------
# Helper: Extract text from a file (PDF, Excel, or Text)
# -----------------------------------------------------------------------------
def extract_text(file):
    filename = file.name.lower()
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif filename.endswith(".xlsx"):
        df_dict = pd.read_excel(file, sheet_name=None)
        text = ""
        for sheet_name, df in df_dict.items():
            text += f"Sheet: {sheet_name}\n"
            text += df.to_csv(index=False) + "\n"
        return text
    else:
        return file.read().decode("utf-8")

# -----------------------------------------------------------------------------
# Admin Console: Upload, Learn, and Refine Calculation Methods
# -----------------------------------------------------------------------------
def admin_console():
    st.header("Admin Console: Upload & Teach Calculation Methods")
    
    uploaded_file = st.file_uploader("Upload a calculation document (PDF, Excel, or Text)", type=["pdf", "xlsx", "txt"])
    explanation = st.text_area("Enter detailed explanation for this method (purpose, steps, assumptions):", height=150)
    
    if st.button("Learn"):
        if not uploaded_file:
            st.error("Please upload a file.")
            return
        if explanation.strip() == "":
            st.error("Please provide a detailed explanation.")
            return
        
        extracted = extract_text(uploaded_file)
        st.subheader("Extracted Content")
        st.text_area("", value=extracted, height=200)
        
        combined = extracted + "\n\nExplanation:\n" + explanation
        st.subheader("Combined Content")
        st.text_area("", value=combined, height=300)
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(combined)
        
        embeddings = OpenAIEmbeddings()
        new_store = FAISS.from_texts(chunks, embeddings)
        
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = new_store
            st.session_state.docs = [combined]
        else:
            st.session_state.docs.append(combined)
            all_chunks = []
            for doc in st.session_state.docs:
                all_chunks.extend(splitter.split_text(doc))
            st.session_state.vectorstore = FAISS.from_texts(all_chunks, embeddings)
        
        st.success("Method learned and added to the knowledge base!")
        
        # Query GPT-4 for alternative methods with chain-of-thought reasoning
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Analyze the following calculation method and suggest any alternative approaches or improvements. "
            "Explain your reasoning step by step before giving your final recommendation:\n\n" + combined
        )
        alt_methods = llm(prompt)
        st.subheader("Alternative Calculation Methods Recommendation")
        st.write(alt_methods)
    
    st.subheader("Feedback & Correction")
    feedback = st.text_area("Enter corrections or clarifications to improve the method:", height=150)
    if st.button("Update Correction"):
        if feedback.strip() != "":
            correction = "Feedback Correction:\n" + feedback
            st.session_state.docs.append(correction)
            
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_chunks = []
            for doc in st.session_state.docs:
                all_chunks.extend(splitter.split_text(doc))
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_texts(all_chunks, embeddings)
            
            st.success("Correction incorporated into the knowledge base!")
        else:
            st.warning("Please provide correction text before updating.")
    
    st.subheader("Export Knowledge Base")
    if st.button("Export to Disk"):
        export_dir = "exported_vectorstore"
        os.makedirs(export_dir, exist_ok=True)
        st.session_state.vectorstore.save_local(export_dir)
        st.success(f"Knowledge base exported to: {export_dir}")

# -----------------------------------------------------------------------------
# Client Chatbot: Ask Questions and Get Expert Answers
# -----------------------------------------------------------------------------
def client_console():
    st.header("Client Chatbot: Ask Your Structural Engineering Question")
    question = st.text_input("Enter your question (e.g., 'How do I calculate lateral wind load on an exterior facade per ASCE 7?')")
    
    if st.button("Submit Question"):
        fallback_llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        expert_instr = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Explain your reasoning step by step and then provide a detailed final answer."
        )
        if st.session_state.vectorstore is None:
            ans = fallback_llm(f"{expert_instr}\n\nQuestion: {question}")
            st.write("**Answer:**")
            st.write(ans)
        else:
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            )
            retrieval_ans = retrieval_chain.run(question)
            # If the retrieval answer is short or seems incomplete, use fallback
            if len(retrieval_ans.strip()) < 50 or "not defined" in retrieval_ans.lower():
                fallback_ans = fallback_llm(f"{expert_instr}\n\nThis question is not fully covered in our approved methods. Please answer: {question}")
                final_ans = ("Our approved calculation methods do not fully cover this question. However, based on broader expert knowledge, here is our best answer: " + fallback_ans)
            else:
                final_ans = retrieval_ans
            st.write("**Answer:**")
            st.write(final_ans)

# -----------------------------------------------------------------------------
# Main: Mode Selection (Admin vs. Client)
# -----------------------------------------------------------------------------
def main():
    st.title("State-of-the-Art Engineering Chatbot")
    st.sidebar.title("Master Console")
    mode = st.sidebar.radio("Select Mode", ["Admin Console", "Client Chatbot"])
    
    if mode == "Admin Console":
        st.write("Welcome to the Admin Console. Upload and teach new calculation methods here.")
        admin_console()
    else:
        st.write("Welcome to the Client Chatbot. Ask your structural engineering questions below.")
        client_console()

if __name__ == '__main__':
    main()
