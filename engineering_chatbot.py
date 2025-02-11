import os
import time
import asyncio
import hashlib
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pytesseract
from openpyxl import load_workbook

# Manually specify the path to your Tesseract executable
# (Update with your actual installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from sympy import sympify, simplify

# Optional Excel evaluator
try:
    from xlcalculator import ModelCompiler, Evaluator
    HAS_XLCALCULATOR = True
except ImportError:
    HAS_XLCALCULATOR = False

# LangChain imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

# Import your config settings
from config import (
    OPENAI_API_KEY,
    VECTOR_STORE_PATH,
    UPLOAD_DIR,
    OCR_TEMP_DIR,
    ADMIN_USERNAME,
    ADMIN_PASSWORD,
    BASE_DIR
)

DEBUG = True
BYPASS_RETRIEVAL = False

# Initialize session state variables if not present
if "admin_upload_content" not in st.session_state:
    st.session_state.admin_upload_content = ""
if "pdf_summary_cache" not in st.session_state:
    st.session_state.pdf_summary_cache = {}

########################
# Helper / Utility
########################

def debug_print(*args):
    if DEBUG:
        print(*args)

def ensure_dirs_exist():
    """Ensure upload directories exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OCR_TEMP_DIR, exist_ok=True)

ensure_dirs_exist()

def read_file_data(uploaded_file):
    """
    Reads bytes from a Streamlit UploadedFile or returns None if invalid.
    Returns (filename, raw_bytes).
    """
    if uploaded_file is None:
        return None, None
    try:
        data = uploaded_file.read()
        name = uploaded_file.name
        return name, data
    except Exception as e:
        debug_print("[DEBUG] Error reading uploaded file:", e)
        return None, None

def get_excel_evaluator(wb):
    try:
        from xlcalculator import ModelCompiler, Evaluator
        data = {}
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            sheet_data = {}
            for row in ws.iter_rows():
                for cell in row:
                    sheet_data[cell.coordinate] = cell.value
            data[sheet] = sheet_data
        compiler = ModelCompiler()
        model = compiler.read_and_parse_dict(data)
        evaluator = Evaluator(model)
        return evaluator
    except Exception as e:
        debug_print("[DEBUG] Excel evaluator error:", e)
        return None

def advanced_understand_formula(formula_string):
    formula_clean = formula_string.lstrip('=').replace('^', '**')
    try:
        expr = sympify(formula_clean)
        simplified_expr = simplify(expr)
        return str(simplified_expr)
    except Exception as e:
        return f"Error in advanced understanding: {e}"

def summarize_text_chunk(text_chunk):
    prompt = (
        "Summarize the following engineering text concisely, "
        "focusing on key approvals, calculation methods, and important notes:\n\n"
        f"{text_chunk}\n\nSummary:"
    )
    messages = [HumanMessage(content=prompt)]
    try:
        llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY, max_tokens=150)
        response = llm(messages)
        return response.content.strip() if response and hasattr(response, "content") else ""
    except Exception as e:
        return f"Error summarizing chunk: {e}"

def advanced_understand_pdf(pdf_text):
    """Use chunked summarization with caching."""
    cache = st.session_state.pdf_summary_cache
    text_hash = hashlib.sha256(pdf_text.encode('utf-8')).hexdigest()
    if text_hash in cache:
        debug_print("[DEBUG] Returning cached PDF summary.")
        return cache[text_hash]

    max_chunk_size = 2000
    if len(pdf_text) <= max_chunk_size:
        summary = summarize_text_chunk(pdf_text)
    else:
        lines = pdf_text.splitlines()
        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 < max_chunk_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        debug_print(f"[DEBUG] Split PDF text into {len(chunks)} chunks.")

        chunk_summaries = []
        for chunk in chunks:
            summary_chunk = summarize_text_chunk(chunk)
            chunk_summaries.append(summary_chunk)

        combined_summary = "\n".join(chunk_summaries)
        summary = summarize_text_chunk(combined_summary)

    cache[text_hash] = summary
    return summary

def extract_excel_content(file_path):
    debug_print("[DEBUG] Extracting Excel content from", file_path)
    wb = load_workbook(file_path, data_only=False)
    evaluator = get_excel_evaluator(wb) if HAS_XLCALCULATOR else None

    content_list = []
    for sheet in wb.worksheets:
        sheet_content = f"Sheet: {sheet.title}\n"
        for row in sheet.iter_rows(values_only=False):
            for cell in row:
                if cell.data_type == 'f':
                    cell_text = f"Cell {cell.coordinate} Formula: {cell.value}"
                    adv = advanced_understand_formula(cell.value)
                    cell_text += f" | Advanced Understanding: {adv}"
                    if evaluator:
                        try:
                            result = evaluator.evaluate(f"{sheet.title}!{cell.coordinate}")
                            cell_text += f" | Evaluated Result: {result}"
                        except Exception as e:
                            cell_text += f" (Eval Error: {str(e)})"
                else:
                    cell_text = f"Cell {cell.coordinate} Value: {cell.value}"
                sheet_content += cell_text + "\n"
        content_list.append(sheet_content)
    return "\n".join(content_list)

def process_pdf_file(file_path):
    """OCR + chunked summarization of PDF."""
    text = ""
    doc = fitz.open(file_path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        page_text = page.get_text().strip()
        if page_text:
            text += page_text + "\n"
        else:
            pix = page.get_pixmap()
            img_path = os.path.join(OCR_TEMP_DIR, f"page_{i}.png")
            pix.save(img_path)
            img = Image.open(img_path)
            text += pytesseract.image_to_string(img) + "\n"
            os.remove(img_path)

    adv_summary = advanced_understand_pdf(text)
    combined = f"Extracted Text:\n{text}\n\nAdvanced PDF Understanding:\n{adv_summary}"
    return combined

##############################
# Admin Functions
##############################

def generate_contextual_outcomes(admin_text, excel_content, pdf_content):
    """
    Combine the data into a single 'File Learning Outcomes' text.
    """
    combined = (
        f"Admin Explanation:\n{admin_text}\n\n"
        f"Excel Content:\n{excel_content}\n\n"
        f"PDF Content:\n{pdf_content}"
    )
    prompt = (
        "You are a specialized engineering AI. Below is new data (Excel, PDF, etc.) plus "
        "the user's explanation. Provide a single 'File Learning Outcome' that includes:\n"
        "1) A concise but detailed interpretation of the data.\n"
        "2) Context-aware questions or clarifications for ambiguous points.\n"
        "3) Any alternative or more efficient methods.\n\n"
        f"{combined}\n\n"
        "Output:\n"
    )
    messages = [HumanMessage(content=prompt)]
    llm = ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm, messages)
            response = future.result(timeout=30)
        if response and hasattr(response, "content"):
            return response.content.strip()
        else:
            return "Error: No response from LLM."
    except Exception as e:
        return f"Error generating outcomes: {str(e)}"

def admin_learn(admin_text, excel_files, pdf_files):
    """Process files, combine data, return a single outcome text."""
    excel_content = ""
    pdf_content = ""

    # Save & Process Excel
    if excel_files:
        for f in excel_files:
            if f is not None:
                fname, fdata = read_file_data(f)
                if fname and fdata:
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    with open(fpath, "wb") as out:
                        out.write(fdata)
                    excel_content += extract_excel_content(fpath) + "\n"
                    os.remove(fpath)

    # Save & Process PDFs
    if pdf_files:
        for f in pdf_files:
            if f is not None:
                fname, fdata = read_file_data(f)
                if fname and fdata:
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    with open(fpath, "wb") as out:
                        out.write(fdata)
                    pdf_content += process_pdf_file(fpath) + "\n"
                    os.remove(fpath)

    outcomes = generate_contextual_outcomes(admin_text, excel_content, pdf_content)
    return outcomes

def confirm_learning(admin_text, excel_files, pdf_files):
    """
    Actually store the new data into the FAISS vector store.
    """
    outcomes = admin_learn(admin_text, excel_files, pdf_files)
    st.session_state.admin_upload_content = outcomes

    embed_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    try:
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embed_model,
                allow_dangerous_deserialization=True
            )
            vector_store.add_texts([outcomes])
        else:
            vector_store = FAISS.from_texts([outcomes], embed_model)
        vector_store.save_local(VECTOR_STORE_PATH)
        return f"Learning confirmed and vector store updated!\n\n{outcomes}"
    except Exception as e:
        return f"Error updating vector store: {str(e)}\n\n{outcomes}"

##############################
# Client Mode
##############################

async def client_chat(query):
    """
    Retrieve from the vector store or fallback to direct call.
    """
    if (BYPASS_RETRIEVAL or
        not os.path.exists(VECTOR_STORE_PATH) or
        not st.session_state.admin_upload_content.strip()):
        # Direct LLM
        prompt = (
            f"You are a specialized engineering bot. User asks:\n{query}\n\n"
            "Provide a detailed, step-by-step answer referencing known standards."
        )
        messages = [HumanMessage(content=prompt)]
        llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
        try:
            response = await asyncio.to_thread(llm, messages)
            return response.content if response else "No response"
        except Exception as e:
            return f"Error in direct call: {str(e)}"
    else:
        # Retrieval QA
        def retrieval_run(q):
            embed_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embed_model,
                allow_dangerous_deserialization=True
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY),
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                verbose=True
            )
            return qa_chain.run(q)

        try:
            response = await asyncio.to_thread(retrieval_run, query)
            return response
        except Exception as e:
            return f"Error in retrieval QA: {str(e)}"

##############################
# MAIN Streamlit UI
##############################

def main():
    st.set_page_config(page_title="Engineering Chatbot", layout="wide")
    st.title("Engineering Chatbot (Streamlit Version)")

    tabs = st.tabs(["Admin Mode (Private)", "Client Mode"])
    # -----------------------------------
    # Admin Mode
    # -----------------------------------
    with tabs[0]:
        st.subheader("Master Console: Upload & Learn")
        admin_text_input = st.text_area(
            "Admin Explanation/Context",
            placeholder="Describe the methods, references, or context here...",
            height=100
        )
        # File Uploaders (accept multiple)
        excel_files = st.file_uploader(
            "Upload Excel Files",
            type=["xlsx","xls"],
            accept_multiple_files=True
        )
        pdf_files = st.file_uploader(
            "Upload PDF Files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process Uploads (Preview)"):
            outcomes = admin_learn(admin_text_input, excel_files, pdf_files)
            st.text_area("File Learning Outcomes", outcomes, height=300)

        if st.button("Confirm and Learn"):
            result = confirm_learning(admin_text_input, excel_files, pdf_files)
            st.text_area("Learning Status", result, height=300)

    # -----------------------------------
    # Client Mode
    # -----------------------------------
    with tabs[1]:
        st.subheader("Client Mode: Ask a Question")
        user_query = st.text_input("Your Query:")
        if "client_chat_history" not in st.session_state:
            st.session_state.client_chat_history = []

        if st.button("Send Query"):
            answer = asyncio.run(client_chat(user_query))
            # Store conversation in session
            st.session_state.client_chat_history.append(("user", user_query))
            st.session_state.client_chat_history.append(("assistant", answer))

        # Display chat history
        for role, content in st.session_state.client_chat_history:
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Chatbot:** {content}")


if __name__ == "__main__":
    main()
