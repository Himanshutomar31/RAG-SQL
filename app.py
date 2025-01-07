import os
import pickle
import streamlit as st
import pyodbc
import faiss
from vector_faiss_db import FAISSHandler
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI


class SQLQueryGenerator:
    def __init__(self, endpoint, deployment, subscription_key):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-05-01-preview",
        )

    def generate_query(self, schema_text, query_text):
        chat_prompt = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps write SQL queries for Microsoft SQL Server.",
            },
            {
                "role": "user",
                "content": f"The database schema is:\n{schema_text}\n\nWrite only the SQL query for the following request. Do not include any explanation, context, or comments:\n{query_text}",
            },
        ]
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=chat_prompt,
            max_tokens=800,
            temperature=0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        return completion.choices[0].message.content


def load_faiss_index_and_texts(index_path, texts_path):
    faiss_index = faiss.read_index(index_path)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
    return faiss_index, texts


def get_sentence_transformer_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return [model.encode([text])[0] for text in texts]


def execute_sql_query(query, server, database, username, password):
    """
    Execute an SQL query against the database and return results.

    Args:
        query (str): The SQL query to execute.
        server (str): Database server address.
        database (str): Database name.
        username (str): Username for authentication.
        password (str): Password for authentication.

    Returns:
        list: Query results as a list of tuples.
    """
    try:
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={username};PWD={password};ConnectTimeout=60;"
        )
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as exp:
        print(str(exp))
        st.error(str(exp))

def main():
    st.title("SQL Query Generator and Executor")
    st.write("This app generates SQL queries based on database schema and user input using Azure OpenAI and FAISS. It also allows executing the generated SQL against a database.")

    # Azure OpenAI credentials
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Database credentials
    server = os.getenv("SERVER")
    database = os.getenv("DATABASE")
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")

    # Load FAISS index and texts
    faiss_index_path = 'faiss_index.index'
    texts_path = 'texts.pkl'
    faiss_index, texts = load_faiss_index_and_texts(faiss_index_path, texts_path)

    # Initialize FAISS handler
    faiss_handler = FAISSHandler(len(texts[0]))
    faiss_handler.index = faiss_index
    faiss_handler.texts = texts

    # Input: User query
    user_query = st.text_input("Enter your SQL query description", "List Products That Have Never Been Ordered")

    if st.button("Generate SQL Query"):
        # Generate embedding for user query
        query_embedding = get_sentence_transformer_embeddings([user_query])[0]

        # Retrieve nearest schema texts
        distances, indices, nearest_texts = faiss_handler.retrieve_embeddings(query_embedding)

        st.subheader("Closest Schema Matches")
        for text in nearest_texts:
            st.text(text)

        # Generate SQL query
        schema_text = "\n".join(nearest_texts)
        sql_generator = SQLQueryGenerator(endpoint, deployment, subscription_key)
        sql_query = sql_generator.generate_query(schema_text, user_query)

        # Display SQL query
        st.subheader("Generated SQL Query")
        st.code(sql_query, language="sql")

        # Execute the SQL query
        try:
            results = execute_sql_query(sql_query, server, database, username, password)
            st.subheader("Query Results")
            for row in results:
                st.write(row)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
