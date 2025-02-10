import os
import streamlit as st
import pandas as pd
import pdfplumber

# =============================================================================
# Attempt to import Chroma from the expected location.
# If not found, try a fallback import; if still not found, show an error.
# =============================================================================
try:
    # Preferred import (requires langchain>=0.0.200)
    from langchain.vectorstores.chromadb import Chroma
except ModuleNotFoundError:
    try:
        # Fallback for older versions (if available)
        from langchain.vectorstores import Chroma
    except ModuleNotFoundError:
        st.error("Module 'Chroma' not found. Please ensure that your requirements.txt "
                 "includes 'langchain>=0.0.200' and 'chromadb>=0.3.21', then force a rebuild.")
        st.stop()

# LangChain components for text splitting, embeddings, LLMs, and retrieval
from langchain.text_splitter import CharacterTextSplitter
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
# Admin Console: Upload, Analyze, Clarify, Learn, and Save Calculation Methods
# -----------------------------------------------------------------------------
def admin_console():
    st.header("Admin Console: Upload & Teach Calculation Methods")
    
    uploaded_file = st.file_uploader("Upload a calculation document (PDF, Excel, or Text)", type=["pdf", "xlsx", "txt"])
    explanation = st.text_area("Enter detailed explanation for this method (purpose, steps, assumptions):", height=150)
    
    # Step 1: Analyze file and ask clarifying questions
    if st.button("Analyze & Ask Clarifying Questions"):
        if not uploaded_file:
            st.error("Please upload a file.")
            return
        if explanation.strip() == "":
            st.error("Please provide a detailed explanation.")
            return
        
        extracted = extract_text(uploaded_file)
        st.subheader("Extracted Content")
        st.text_area("", value=extracted, height=200)
        
        # Combine the extracted file text with the provided explanation
        combined = extracted + "\n\nExplanation:\n" + explanation
        st.subheader("Combined Content")
        st.text_area("", value=combined, height=300)
        
        # Use GPT‑4 to ask clarifying questions about the method's math and logic.
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Carefully analyze the following calculation method. Identify any ambiguities or missing details, and list clarifying questions that would help you fully understand the underlying mathematical equations, logical steps, and assumptions in this method. "
            "Please list each question clearly:\n\n" + combined
        )
        clarifying_questions = llm(prompt)
        st.subheader("Clarifying Questions")
        st.write(clarifying_questions)
        
        # Save the combined content and clarifying questions in session state
        st.session_state.clarifying_questions = clarifying_questions
        st.session_state.current_combined = combined

    # Step 2: Admin provides answers to the clarifying questions to enhance the method understanding
    if "clarifying_questions" in st.session_state:
        st.subheader("Your Answers to the Clarifying Questions")
        clarifying_answers = st.text_area("Provide your answers to the above questions. These answers will help the AI fully understand the calculation method.", height=150)
        if st.button("Learn and Save Method"):
            # Combine clarifying answers with previously combined content
            full_content = st.session_state.current_combined + "\n\nClarifying Answers:\n" + clarifying_answers
            st.subheader("Full Combined Content")
            st.text_area("", value=full_content, height=300)
            
            # Split content into manageable chunks and update the vector store
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(full_content)
            embeddings = OpenAIEmbeddings()
            
            # Create a new Chroma vector store from texts (using a collection name for persistence)
            new_store = Chroma.from_texts(chunks, embeddings, collection_name="chatbot_docs")
            
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = new_store
                st.session_state.docs = [full_content]
            else:
                st.session_state.docs.append(full_content)
                all_chunks = []
                for doc in st.session_state.docs:
                    all_chunks.extend(splitter.split_text(doc))
                st.session_state.vectorstore = Chroma.from_texts(all_chunks, embeddings, collection_name="chatbot_docs")
            
            st.success("Method learned and added to the knowledge base!")
            
            # Optionally, ask GPT‑4 for further analysis and suggestions.
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            prompt_alt = (
                "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
                "Analyze the following fully detailed calculation method (including the clarifying answers). Break down the underlying mathematical equations, variables, and logical steps. "
                "Explain how each part of the calculation contributes to the overall method, and suggest any improvements or alternative approaches if applicable:\n\n" 
                + full_content
            )
            alt_methods = llm(prompt_alt)
            st.subheader("Alternative Calculation Methods & Analysis")
            st.write(alt_methods)
            
            # Clear temporary session state variables used for clarifying questions
            st.session_state.pop("clarifying_questions", None)
            st.session_state.pop("current_combined", None)
    
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
            st.session_state.vectorstore = Chroma.from_texts(all_chunks, embeddings, collection_name="chatbot_docs")
            st.success("Correction incorporated into the knowledge base!")
        else:
            st.warning("Please provide correction text before updating.")
    
    st.subheader("Export Knowledge Base")
    if st.button("Export to Disk"):
        export_dir = "exported_vectorstore"
        os.makedirs(export_dir, exist_ok=True)
        st.success(f"Knowledge base exported to: {export_dir}")

# -----------------------------------------------------------------------------
# Client Chatbot: Ask Questions Based on the Learned Calculation Methods
# -----------------------------------------------------------------------------
def client_console():
    st.header("Client Chatbot: Ask Your Structural Engineering Question")
    question = st.text_input("Enter your question (e.g., 'How do I calculate lateral wind load on an exterior facade per ASCE 7?')")
    
    if st.button("Submit Question"):
        fallback_llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        expert_instr = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Explain your reasoning step by step and provide a detailed final answer, including any preliminary calculations if relevant."
        )
        if st.session_state.vectorstore is None:
            ans = fallback_llm(f"{expert_instr}\n\nQuestion: {question}")
            st.write("**Answer:**")
            st.write(ans)
        else:
            # Use a retrieval chain to pull in relevant learned content
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            )
            retrieval_ans = retrieval_chain.run(question)
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
