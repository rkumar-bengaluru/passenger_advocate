
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
import os 

class BaseEmbedding(ABC):
    def __init__(self, model: Optional[str] = None, *args, **kwargs):
        self.name = model 
        self.model = self.load_model()

    @abstractmethod
    def load_model(self, *args, **kwargs) -> "BaseEmbedding":
        pass

    @abstractmethod
    def create_embedding(self):
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        pass 

    def get_model_name(self, *args, **kwargs) -> str:
        return self.name
    


class OpenAIEmbeddingModel(BaseEmbedding):

    def __init__(
        self,
        file: Optional[str] = "data/policies.md",
        collection_name: Optional[str] = "policy_embeddings",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.file = file 
        
        if api_key:
            self.api_key = SecretStr(api_key)
        else:
            env_key = os.getenv("OPENAI_API_KEY")
            if not env_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.api_key = SecretStr(env_key)

        if model:
            self.model = model 
        else:
            self.model = os.getenv("OPENAI_EMBEDDING_MODEL")

        self.chunk_size = float(os.getenv("OPENAI_CHUNK_SIZE", 1024))
        self.chunk_over_lap = float(os.getenv("OPENAI_CHUNK_OEVRLAP", 1024))
        self.dimension = float(os.getenv("OPENAI_EMBEDDING_DIMENSION", 1536))

        if os.getenv("QDRANT_CONNECTION_STRING"):
            self.qdrant_connection_string =  os.getenv("QDRANT_CONNECTION_STRING")
        else:
            raise ValueError("QDRANT_CONNECTION_STRING not found in environment")
        
        if os.getenv("QDRANT_API_KEY"):
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        else:
            raise ValueError("QDRANT_API_KEY not found in environment")
        
        self.collection_name = collection_name
        self.client: Optional[OpenAIEmbeddings] = None
        self.qdrant_client: Optional[QdrantClient] = None

    def load_model(self) -> OpenAIEmbeddings:
        if self.client is None:
            self.client = OpenAIEmbeddings(
                model=self.model,   # or text-embedding-3-small
                api_key=self.api_key.get_secret_value()
            )
        if self.qdrant_client is None:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_connection_string,  
                api_key=self.qdrant_api_key,
                timeout = 60,
                check_compatibility=True,
                prefer_grpc=True,
            )

        return self.client
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        """
        Search similar documents in Qdrant
        
        Args:
            query (str): User query
            top_k (int): Number of results to return
            score_threshold (float): Minimum similarity score (0.0 to 1.0)
        
        Returns:
            List of dicts with text, score, and metadata
        """
        try:
            client = self.load_model()
            # Initialize vector store
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=client
            )

            # Perform similarity search
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                score_threshold=score_threshold
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })

            print(f"Found {len(formatted_results)} relevant chunks for query.")
            return formatted_results

        except Exception as e:
            print(f"Error during search: {type(e).__name__}: {e}")
            return []
        
    def create_embedding(self):
        client = self.load_model()  # assuming this returns an Embeddings instance

        # 1. Load policy text
        with open(self.file, "r", encoding="utf-8") as f:
            policy_text = f.read()

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_over_lap,
            length_function=len
        )
        chunks = text_splitter.split_text(policy_text)

        collection_name = self.collection_name  # use consistently

        # Check if collection exists
        try:
            collections = self.qdrant_client.get_collections().collections
            existing = any(c.name == collection_name for c in collections)
        except Exception as e:
            print(f"Error fetching collections: {e}")
            existing = False

        if existing:
            print(f"Collection '{collection_name}' already exists.")
            # Optionally: you can still add more texts if needed
            # vector_store = QdrantVectorStore.from_existing_collection(...)
            return

        print(f"Collection '{collection_name}' does not exist. Creating...")

        # Better approach: Use from_texts (creates collection + adds data in one go)
        try:
            vector_store = QdrantVectorStore.from_texts(
                texts=chunks,
                embedding=client,
                client=self.qdrant_client,
                collection_name=collection_name,
                # Optional but recommended:
                distance_func="Cosine",  # or "Dot", "Euclid"
            )
            print(f"Successfully created collection '{collection_name}' and inserted {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            # Fallback: manual creation + add_texts
            self._create_collection_manually(collection_name, client)
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=client
            )
            vector_store.add_texts(texts=chunks)


    def _create_collection_manually(self, collection_name: str, embedding):
        try:
            # Test connection first
            self.qdrant_client.get_collections()
            print("✓ Connected to Qdrant successfully")
            
            # Get embedding dimension
            dimension = len(embedding.embed_query("test sentence"))
            print(f"Embedding dimension: {dimension}")
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,   # or EUCLID / DOT
                ),
            )
            print(f"✓ Collection '{collection_name}' created successfully.")
            
        except Exception as e:
            print(f"❌ Error creating collection: {type(e).__name__}: {e}")
            raise
        