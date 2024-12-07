# AI-Powered Text-to-SQL Generator

This project is an **AI-powered Text-to-SQL generator** that converts natural language queries into SQL queries using **GPT-4** and **FAISS** vector databases. By leveraging **Retrieval-Augmented Generation (RAG)**, the tool intelligently retrieves relevant database schema information to generate accurate SQL queries.

## Features

- **GPT-4 Integration**: Utilizes OpenAI's GPT-4 model to generate SQL queries from natural language input.
- **FAISS Vector Database**: Stores and retrieves database schema information using FAISS, enabling efficient semantic search.
- **Retrieval-Augmented Generation (RAG)**: Combines the power of semantic search and GPT-4 to improve query generation.
- **Customizable**: Easily configurable to work with different SQL database schemas.
- **Future Web Tool**: Plans to create a web-based interface for ease of use.

## Prerequisites

- Python 3.7+
- OpenAI API Key (for GPT-4 access)
- FAISS library for vector database handling

## How It Works

1. **Schema Embeddings**: The tool connects to your SQL database, retrieves the schema, and generates embeddings using the **Sentence-Transformer** model or **OpenAI GPT-4**.
2. **Query Generation**: The user provides a natural language query, which is processed using **RAG** and FAISS to retrieve relevant schema details, and then GPT-4 generates the corresponding SQL query.

3. **Vector Database**: FAISS is used for storing and retrieving embeddings of the schema for efficient search.

## Example Usage

To generate a SQL query for "Find users with orders above a certain total", the system retrieves relevant schema information and uses GPT-4 to generate the query:

```sql
SELECT ProductID, ProductName
FROM Products
WHERE ProductID NOT IN (SELECT ProductID FROM OrderDetails)
```
