import os
import pyodbc
import openai
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from vector_faiss_db import FAISSHandler


class SQLSchemaHandler:
    """
    Handles interactions with the SQL Server database and retrieves schema information.
    """

    def __init__(self, server, database, username, password):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={username};PWD={password}"
        )

    def get_schema_info(self):
        """
        Fetches schema information from the database.

        Returns:
            dict: A dictionary containing table names as keys and column details as values.
        """
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
        """)
        tables = cursor.fetchall()

        schema_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
            """, table_name)
            columns = cursor.fetchall()

            schema_info[table_name] = [
                {"column_name": column[0], "data_type": column[1]} for column in columns
            ]

        conn.close()
        return schema_info


class SchemaTextConverter:
    """
    Converts database schema information into a text format for embedding generation.
    """

    @staticmethod
    def convert_schema_to_text(schema_info):
        """
        Converts schema info into a human-readable text format.

        Args:
            schema_info (dict): Schema information.

        Returns:
            list: A list of text descriptions for each table.
        """
        schema_text = []
        for table_name, columns in schema_info.items():
            table_str = f"Table: {table_name}\n"
            for col in columns:
                table_str += f"  Column: {col['column_name']}, Data Type: {col['data_type']}\n"
            schema_text.append(table_str)
        return schema_text


class EmbeddingGenerator:
    """
    Generates embeddings for text using various embedding models.
    """

    def __init__(self, use_openai=False):
        self.use_openai = use_openai
        self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2') if not use_openai else None

    def get_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list): List of strings to embed.

        Returns:
            list: List of embedding vectors.
        """
        if self.use_openai:
            return [self._get_openai_embedding(text) for text in texts]
        else:
            return [self.sentence_transformer_model.encode([text])[0] for text in texts]

    @staticmethod
    def _get_openai_embedding(text):
        """
        Generates embeddings using OpenAI API.

        Args:
            text (str): Input text.

        Returns:
            list: Embedding vector.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']


class SchemaEmbeddingPipeline:
    """
    Manages the entire process of fetching schema, converting to text, and generating embeddings.
    """

    def __init__(self, server, database, username, password, use_openai=False):
        self.schema_handler = SQLSchemaHandler(server, database, username, password)
        self.text_converter = SchemaTextConverter()
        self.embedding_generator = EmbeddingGenerator(use_openai)

    def generate_embeddings(self):
        """
        Generates schema embeddings.

        Returns:
            tuple: Embeddings and corresponding texts.
        """
        schema_info = self.schema_handler.get_schema_info()
        schema_text = self.text_converter.convert_schema_to_text(schema_info)
        embeddings = self.embedding_generator.get_embeddings(schema_text)
        return embeddings, schema_text


def main():
    # Fetch credentials from environment variables
    server = os.getenv("SERVER")
    database = os.getenv("DATABASE")
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")

    # Initialize the pipeline
    pipeline = SchemaEmbeddingPipeline(server, database, username, password, use_openai=False)

    # Generate embeddings and text
    embeddings, texts = pipeline.generate_embeddings()

    # Initialize FAISS handler
    faiss_handler = FAISSHandler(len(embeddings[0]))
    faiss_handler.store_embeddings(embeddings, texts)

    # Save FAISS index and corresponding texts
    faiss.write_index(faiss_handler.index, 'faiss_index.index')
    with open('texts.pkl', 'wb') as f:
        pickle.dump(faiss_handler.texts, f)


if __name__ == "__main__":
    main()
