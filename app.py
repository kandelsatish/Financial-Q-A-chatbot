import os
import re
import time
import tempfile
import warnings
import logging
import traceback
import pickle as pkl
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from llama_parse import LlamaParse
from groq import Groq
import streamlit as st
from table_extract import *
from rag_groq import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the rf vectorizer and classifier model
with open('/home/bijay/capstone_project/Financial-Q-A-chatbot/models/rf_tfidfvectorizer.pkl', 'rb') as f:
    vectorizer = pkl.load(f)
with open('/home/bijay/capstone_project/Financial-Q-A-chatbot/models/rf_classifer_model.pkl', 'rb') as f:
    model = pkl.load(f)

# Suppress warnings and configure logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)

# Load environment variables
load_dotenv()

# Retrieve necessary variables from the environment
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client and database connection
groq_client = Groq(api_key=groq_api_key)
connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)



def empty_arithmetic_questions_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP DATABASE arithmetic_questions"))
            conn.execute(text("CREATE DATABASE arithmetic_questions"))
        st.success("Arithmetic questions database emptied successfully.")
    except Exception as e:
        st.error(f"Error emptying arithmetic questions database: {str(e)}")
        st.error(traceback.format_exc())


def process_user_question(question, context, question_type):
    similar_text = get_similar_sentences(question, context)
    relevant_context = ""
    for sentence, similarity in similar_text:
        relevant_context += sentence + " "
    if question_type == 'arithmetic':
        answer = generate_answer_from_table_context(question, relevant_context)
    elif question_type == 'span':
        answer = generate_answer_from_text_context(question, relevant_context)
    return answer


def main():
    # Set page configuration
    st.set_page_config(page_title="Financial Q&A Chatbot",page_icon="💵",layout="centered")
    st.title("Welcome to Financial Q&A Chatbot 💵🤖")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        try:
            empty_arithmetic_questions_db()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            try:
                documents = LlamaParse(api_key=llama_cloud_api_key, result_type="markdown").load_data(tmp_file_path)
                text_documents = [doc.text for doc in documents]
                text_context = extract_text_from_markdown('\n'.join(text_documents))
                # text_context = extract_text_from_markdown('\n'.join(documents))
                print("length of text_context", len(text_context))
                with engine.connect() as conn:
                    conn.execute(text("USE arithmetic_questions"))
                    for i, doc in enumerate(documents):
                        lowercase_markdown = doc.text.lower()
                        tables = extract_tables_from_markdown(lowercase_markdown)
                        for j, table in enumerate(tables):
                            table_name = f"pdf_table_{i+1}_{j+1}"
                            # Check for blank column names
                            if any(col == '' or col is None for col in table.columns):
                                print("Invalid column names detected, assigning default names.")
                                table.columns = get_unique_column_names(table)
                            try:
                                table.to_sql(table_name, engine, if_exists='replace', index=False)
                            except Exception as e:
                                st.error(f"Error inserting table {j+1} from document {i+1}: {str(e)}")
                                st.error(traceback.format_exc())
                st.success("PDF processing completed.")
                table_context = prepare_context(engine)
            finally:
                os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")
            st.error(traceback.format_exc())

    st.write("Enter your question below:")
    question_input = st.text_input("Question")
    if st.button("Find Answer"):
        if question_input:
            try:
                pred_q = vectorizer.transform([question_input])
                question_type = model.predict(pred_q)[0]
                if question_type == 'arithmetic':
                    st.write("question type: Arithmetic")
                    result = process_user_question(question_input, table_context, question_type)
                else:
                    st.write("question type: Span")
                    result = process_user_question(question_input, text_context, question_type)
                if result is not None:
                    st.write(result)
                else:
                    st.warning("No results found or an error occurred during query execution.")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
                st.error(traceback.format_exc())
        else:
            st.warning("Please enter a question before clicking 'Find Answer'.")

if __name__ == "__main__":
    main()