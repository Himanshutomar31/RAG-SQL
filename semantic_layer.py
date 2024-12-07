import os
import pickle
import faiss
from vector_faiss_db import FAISSHandler
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI


class SQLQueryGenerator:
    """
    A class to generate SQL queries based on database schema and user input.
    """

    def __init__(self, endpoint, deployment, subscription_key):
        """
        Initialize the Azure OpenAI client for SQL query generation.

        Args:
            endpoint (str): Azure OpenAI API endpoint URL.
            deployment (str): Deployment name for the Azure OpenAI service.
            subscription_key (str): API key for the Azure OpenAI service.
        """
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-05-01-preview",
        )

    def generate_query(self, schema_text, query_text):
        """
        Generate an SQL query based on the schema and the query request.

        Args:
            schema_text (str): The database schema as a text description.
            query_text (str): The user query describing the SQL request.

        Returns:
            str: The generated SQL query.
        """
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
    """
    Load the FAISS index and corresponding texts from disk.

    Args:
        index_path (str): Path to the FAISS index file.
        texts_path (str): Path to the file containing corresponding texts.

    Returns:
        tuple: (faiss_index, texts)
    """
    faiss_index = faiss.read_index(index_path)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
    return faiss_index, texts


def get_sentence_transformer_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = []
    for text in texts:
        embeddings.append(model.encode([text])[0])
    return embeddings

def main():
    # Environment variables for Azure OpenAI
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Load FAISS index and texts
    faiss_index_path = 'faiss_index.index'
    texts_path = 'texts.pkl'
    faiss_index, texts = load_faiss_index_and_texts(faiss_index_path, texts_path)

    # Initialize FAISS handler
    faiss_handler = FAISSHandler(len(texts[0]))  # Assuming all embeddings have the same dimension
    faiss_handler.index = faiss_index
    faiss_handler.texts = texts

    # Query embedding and nearest neighbors
    query_text = "List Products That Have Never Been Ordered"
    query_embedding = get_sentence_transformer_embeddings([query_text])[0]
    distances, indices, nearest_texts = faiss_handler.retrieve_embeddings(query_embedding)

    # Display nearest texts
    print(f"Distances: {distances}\nIndices: {indices}")
    print("Nearest Texts:", nearest_texts)

    # Generate SQL query
    schema_text = "\n".join(nearest_texts)
    sql_generator = SQLQueryGenerator(endpoint, deployment, subscription_key)
    sql_query = sql_generator.generate_query(schema_text, query_text)

    # Display generated SQL query
    print("Generated SQL Query:")
    print(sql_query)


if __name__ == "__main__":
    main()
