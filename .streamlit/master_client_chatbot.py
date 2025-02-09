"""
Master & Client Chatbot for PhD-Level Structural Engineering Exterior Facade Calculations
------------------------------------------------------------------------------------------
This application provides two interfaces:

1. Admin Console (Master):
   - Upload trusted calculation documents (PDF, Excel, or text) and provide a detailed explanation.
   - Click "Learn" to extract the text, combine it with your explanation, and update the knowledge base.
   - The system generates embeddings using OpenAI’s API and stores the approved content in a FAISS vector store.
   - It also uses GPT‑4 (with chain-of-thought prompts) to suggest alternative methods and accepts corrections.

2. Client Chatbot:
   - Clients ask questions about exterior façade structural engineering.
   - The system retrieves relevant information from the approved knowledge base and uses GPT‑4 to generate an expert-level answer.
   - If the answer is incomplete, the chatbot falls back to a full GPT‑4 response.

Ensure your OPENAI_API_KEY is set (via the .streamlit secrets file).
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
# Check for the API Key and Initialize Session State
# -----------------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OPENAI_API_KEY in the .streamlit/secrets.toml file.")
    st.stop()

# Initialize the shared FAISS vector store (our knowledge base) and the list of approved documents
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# -----------------------------------------------------------------------------
# Helper Function: Extract Text from an Uploaded File (PDF, Excel, or Plain Text)
# -----------------------------------------------------------------------------
def extract_text(file):
    file_name = file.name.lower()
    if file_name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif file_name.endswith(".xlsx"):
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
    st.header("Admin Console: Upload & Teach New Calculation Methods")

    uploaded_file = st.file_uploader(
        "Upload a calculation document (PDF, Excel, or Text)",
        type=["pdf", "xlsx", "txt"],
    )

    explanation = st.text_area(
        "Enter a detailed explanation for this calculation method (purpose, steps, assumptions, etc.):",
        height=150,
    )

    if st.button("Learn"):
        if not uploaded_file:
            st.error("Please upload a file containing the calculation method.")
            return
        if explanation.strip() == "":
            st.error("Please provide a detailed explanation for the calculation method.")
            return

        extracted_text = extract_text(uploaded_file)
        st.write("**Extracted Document Content:**")
        st.text_area("", value=extracted_text, height=200)

        combined_text = extracted_text + "\n\nExplanation:\n" + explanation
        st.text_area("Combined Content to Learn:", value=combined_text, height=300)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(combined_text)

        embeddings = OpenAIEmbeddings()
        new_vectorstore = FAISS.from_texts(chunks, embeddings)

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = new_vectorstore
            st.session_state.docs = [combined_text]
        else:
            st.session_state.docs.append(combined_text)
            all_texts = []
            for doc in st.session_state.docs:
                all_texts.extend(text_splitter.split_text(doc))
            st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings)

        st.success("New calculation method learned and added to the knowledge base!")

        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        cot_instruction = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Please analyze the following calculation method in detail and suggest alternative approaches or improvements, explaining your reasoning step by step."
        )
        prompt = f"{cot_instruction}\n\n{combined_text}"
        alternative_methods = llm(prompt)
        st.subheader("Alternative Calculation Methods Recommendation:")
        st.write(alternative_methods)

    st.subheader("Feedback & Correction")
    feedback = st.text_area(
        "If the AI made mistakes or if you want to refine the method, enter your corrections or clarifications here:",
        height=150,
    )
    if st.button("Update Correction"):
        if feedback.strip() != "":
            correction_text = "Feedback Correction:\n" + feedback
            st.session_state.docs.append(correction_text)

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts = []
            for doc in st.session_state.docs:
                all_texts.extend(text_splitter.split_text(doc))
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings)

            st.success("Your correction has been incorporated into the knowledge base!")
        else:
            st.warning("Please enter some correction text before updating.")

    st.subheader("Export Knowledge Base")
    if st.button("Export Knowledge Base to Disk"):
        export_dir = "exported_vectorstore"
        os.makedirs(export_dir, exist_ok=True)
        st.session_state.vectorstore.save_local(export_dir)
        st.success(f"Knowledge base exported to: {export_dir}")

# -----------------------------------------------------------------------------
# Client Chatbot: Answer Engineering Questions with Internal Reasoning & Fallback
# -----------------------------------------------------------------------------
def client_console():
    st.header("Client Chatbot: Ask Your Structural Engineering Question")
    question = st.text_input(
        "Enter your question (e.g., 'How do I calculate the lateral wind load on an exterior facade per ASCE 7?')"
    )

    if st.button("Submit Question"):
        fallback_llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        expert_instruction = (
            "You are a PhD-level expert in structural engineering and advanced mathematics specializing in exterior facade design. "
            "Explain your reasoning step by step, and then provide a detailed final answer."
        )

        if st.session_state.vectorstore is None:
            fallback_answer = fallback_llm(f"{expert_instruction}\n\nQuestion: {question}")
            st.write("**Answer:**")
            st.write(fallback_answer)
        else:
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
            )
            retrieval_answer = retrieval_chain.run(question)

            if len(retrieval_answer.strip()) < 50 or "not defined" in retrieval_answer.lower():
                fallback_answer = fallback_llm(
                    f"{expert_instruction}\n\nThis question is not fully covered in our approved methods. "
                    f"Based on your expertise, please answer the following question: {question}"
                )
                final_answer = (
                    "Our approved calculation methods do not completely cover this question. "
                    "However, based on broader expert knowledge, here is our best answer: " + fallback_answer
                )
            else:
                final_answer = retrieval_answer

            st.write("**Answer:**")
            st.write(final_answer)

# -----------------------------------------------------------------------------
# Main: Toggle Between Admin Console and Client Chatbot Modes
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("Master Console")
    mode = st.sidebar.radio("Select Mode", ["Admin Console", "Client Chatbot"])

    if mode == "Admin Console":
        admin_console()
    else:
        client_console()

if __name__ == '__main__':
    main()
