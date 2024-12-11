import os
import re
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import create_engine, text
from llama_parse import LlamaParse
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame
# Load environment variables
load_dotenv()

# Retrieve necessary variables
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Spark Dataframes") \
    .getOrCreate()


# Set up database connection
connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)

# def table_to_text(df):
#     print("df type: ",type(df))
#     column_names = df.columns.tolist()
#     rows = df.values.tolist()
#     formatted_text = ""
#     for row in rows:
#         for head, cell in zip(column_names[1:], row[1:]):
#             if cell and not pd.isna(cell):
#                 formatted_text += f"The {column_names[0]}:{row[0]} of {head} is {cell}. "
#     return formatted_text

from pyspark.sql import DataFrame as SparkDataFrame

def table_to_text(df: SparkDataFrame):
    
    # Get column names
    column_names = df.columns  # This is a list of column names
    
    # Collect rows as a list of Row objects
    rows = df.collect()  # Collect rows from the DataFrame
    
    formatted_text = ""
    
    for row in rows:
        # Convert Row object to a dictionary
        row_dict = row.asDict()
        
        # Iterate over the columns, skipping the first one if needed
        for head in column_names[1:]:
            cell = row_dict[head]
            if cell is not None:  # Check for None instead of NaN
                formatted_text += f"The {column_names[0]}: {row_dict[column_names[0]]} of {head} is {cell}. "
    
    return formatted_text

def extract_table_from_db(table_name, engine):
    with engine.connect() as conn:
        conn.execute(text(f"USE {database}"))
        pandas_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)\
        
    # Convert all columns to string type
    for col in pandas_df.columns:
        pandas_df[col] = pandas_df[col].astype(str)
    
    # Initialize SparkSession if not already done
    spark = SparkSession.builder.getOrCreate()
    
    # Convert pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)
    
    return spark_df

def prepare_context(engine):
    context = ""
    with engine.connect() as conn:
        conn.execute(text(f"USE {database}"))
        tables = conn.execute(text("SHOW TABLES")).fetchall()
        table_names = [table[0] for table in tables]
        for table_name in table_names:
            df = extract_table_from_db(table_name, engine)
            df.name = table_name
            context += table_to_text(df) + "\n\n"
    return context

def generate_answer_from_table_context(question, context):
    prompt = f"""
    Given the following context:
    {context}
    Please answer the following question:
    {question}
   You are a highly skilled mathematical assistant with access to contextual information. Your task is to answer questions accurately and concisely based on the provided context. Follow these steps:

1. Word Matching:
   - Perform strict word matching between the context sentences and the question.
   - Identify sentences with high word match as they are likely most relevant to the question.

2. Question Analysis:
   - Carefully analyze the question to understand what is being asked.
   - Determine if any mathematical operations are required to answer the question.

3. Context Utilization:
   - Focus on the most relevant parts of the context based on word matching and question analysis.
   - Extract only the information necessary to answer the question.

4. Mathematical Processing:
   - If required, perform the necessary mathematical operations using the relevant information from the context.
   - Ensure calculations are accurate and appropriate for the question.

5. Answer Formulation:
   - Construct a clear and concise answer using only the relevant information and results.
   - Avoid including extraneous details not directly related to the question.

6. Final Response:
   - Provide your answer in a concise manner.
   - End with a complete sentence summarizing the answer, formatted in bold.

Remember: Prioritize accuracy, relevance, and conciseness in your responses. Only include information that directly contributes to answering the question at hand.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {"role": "system", "content": "You are a mathematics assistant that answers questions based on context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def generate_answer_from_text_context(question, context):
    prompt = f"""
    Given the following context:
    {context}
    Please answer the following question:
    {question}
    Provide a short and accurate answer as possible from context based on question.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {"role": "system", "content": "You are a answer finder assistant that answers questions based on context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def get_similar_sentences(question, context, top_n=5):
    sentences = re.split(r'\.\s+', context)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question] + sentences)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [(sentences[i], cosine_similarities[i]) for i in top_indices]