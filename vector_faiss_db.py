import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSHandler:
    """
    A handler for managing FAISS indexing and querying for embeddings.
    """

    def __init__(self, dimension):
        """
        Initializes a FAISS index with the specified embedding dimension.

        Args:
            dimension (int): The dimension of the embeddings to be stored.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)  
        self.texts = [] 

    def store_embeddings(self, embeddings, texts):
        """
        Stores embeddings and their corresponding texts in the FAISS index.

        Args:
            embeddings (list or np.ndarray): List of embeddings or a 2D numpy array of embeddings.
            texts (list): List of texts corresponding to the embeddings.
        """
        embeddings_np = np.array(embeddings, dtype=np.float32)  
        self.index.add(embeddings_np)  
        self.texts.extend(texts) 

    def retrieve_embeddings(self, query_embedding, k=3):
        """
        Retrieves the closest embeddings and their corresponding texts from the FAISS index.

        Args:
            query_embedding (list or np.ndarray): The query embedding.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 3.

        Returns:
            tuple: (distances, indices, texts)
                distances (np.ndarray): Distances of the nearest neighbors.
                indices (np.ndarray): Indices of the nearest neighbors.
                texts (list): Corresponding texts of the nearest neighbors.
        """
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding_np, k)  
        nearest_texts = [self.texts[i] for i in indices[0] if i < len(self.texts)]  
        return distances, indices, nearest_texts

    def get_index_info(self):
        """
        Retrieves information about the FAISS index, including the number of embeddings stored.

        Returns:
            int: The number of embeddings currently stored in the index.
        """
        return self.index.ntotal
