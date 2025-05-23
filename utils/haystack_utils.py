# haystack_utils.py
import os
import torch # For Qwen LLM dtype

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.dataclasses import Document # Ensure Document is imported

from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig # Uncomment if using quantization for Qwen

def initialize_document_store():
    """Initializes and returns an InMemoryDocumentStore."""
    document_store = InMemoryDocumentStore()
    print("Initialized InMemoryDocumentStore for Haystack 2.x.")
    return document_store

def embed_and_write_documents(document_store: InMemoryDocumentStore,
                              raw_documents: list, # These are Haystack Document objects
                              embedding_model_name: str,
                              device: str):
    """
    Initializes document embedder, embeds documents, and writes them to the document store.
    """
    if not raw_documents:
        print("No documents provided to embed_and_write_documents. Skipping.")
        return False

    print(f"Initializing SentenceTransformersDocumentEmbedder with model: {embedding_model_name} on device: {device}")
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=embedding_model_name,
        device=device,
    )
    document_embedder.warm_up()
    print("Document embedder warmed up.")

    print(f"Embedding {len(raw_documents)} documents (this may take a while)...")
    try:
        embedding_results = document_embedder.run(documents=raw_documents)
        documents_with_embeddings = embedding_results["documents"]
    except Exception as e:
        print(f"Error during document embedding: {e}")
        return False

    writer = DocumentWriter(document_store=document_store)
    try:
        writer.run(documents=documents_with_embeddings)
        print(f"Written {len(documents_with_embeddings)} documents with embeddings to the InMemoryDocumentStore.")
        print(f"Document count in store: {document_store.count_documents()}")

        all_docs_in_store = document_store.filter_documents()
        if all_docs_in_store and all_docs_in_store[0].embedding is not None:
            print(f"First document in store has an embedding of dimension: {len(all_docs_in_store[0].embedding)}")
        elif all_docs_in_store:
            print("First document in store does NOT have an embedding. Check embedder.")
        else:
            print("Store is empty after writing attempt.")
        return True
    except Exception as e:
        print(f"Error writing documents to store: {e}")
        return False


def initialize_retrievers(document_store: InMemoryDocumentStore,
                          embedding_model_name: str,
                          device: str,
                          top_k_bm25: int,
                          top_k_embedding: int):
    """Initializes BM25 and Embedding retrievers."""
    if document_store.count_documents() == 0:
        print("Document store is empty. Skipping retriever initialization.")
        return None, None, None

    bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k_bm25)
    print(f"InMemoryBM25Retriever initialized with top_k={top_k_bm25}.")

    print(f"Initializing QueryTextEmbedder with model: {embedding_model_name} on device: {device}")
    query_text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model_name,
        device=device
    )
    query_text_embedder.warm_up()
    print("QueryTextEmbedder warmed up.")

    embedding_retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k_embedding
    )
    print(f"InMemoryEmbeddingRetriever initialized with top_k={top_k_embedding}.")

    return bm25_retriever, embedding_retriever, query_text_embedder

def load_qwen_llm(model_name: str, device_map="auto"):
    """Loads the Qwen LLM and its tokenizer."""
    print(f"\nLoading Qwen LLM: {model_name}...")
    print("This may take a while and require significant RAM/VRAM.")
    try:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        qwen_model.eval()
        print("Qwen LLM and Tokenizer loaded successfully.")
        return qwen_model, qwen_tokenizer
    except Exception as e:
        print(f"Error loading Qwen LLM: {e}")
        return None, None

# --- NEWLY ADDED FUNCTION from User's "Cell 9" ---
def get_hybrid_retrieved_documents_hs2(query: str,
                                    bm25_retriever_node: InMemoryBM25Retriever,
                                    embedding_retriever_node: InMemoryEmbeddingRetriever,
                                    query_embedder_node: SentenceTransformersTextEmbedder,
                                    top_k_join: int) -> list[Document]:
    """Performs hybrid retrieval using Haystack 2.x components."""
    if not bm25_retriever_node or not embedding_retriever_node or not query_embedder_node:
        print("Warning: One or more retrieval components are not initialized for hybrid retrieval.")
        return []

    # BM25 Retrieval
    bm25_docs = []
    try:
        bm25_results = bm25_retriever_node.run(query=query)
        bm25_docs = bm25_results.get("documents", [])
        for doc in bm25_docs:
            doc.meta['retrieval_type'] = 'bm25'
            doc.meta['bm25_score'] = doc.score # Store original BM25 score
    except Exception as e:
        print(f"Error during BM25 retrieval: {e}")

    # Embedding Retrieval
    embedding_docs = []
    try:
        # 1. Embed the query
        query_embedding_result = query_embedder_node.run(text=query)
        query_embedding = query_embedding_result["embedding"]

        # 2. Retrieve using the query embedding
        embedding_results = embedding_retriever_node.run(query_embedding=query_embedding)
        embedding_docs = embedding_results.get("documents", [])
        for doc in embedding_docs:
            doc.meta['retrieval_type'] = 'embedding'
            doc.meta['embedding_score'] = doc.score # Store original embedding score
    except Exception as e:
        print(f"Error during embedding retrieval: {e}")

    # Combine and de-duplicate
    all_docs_dict = {}
    # Add BM25 docs, storing their scores
    for doc in bm25_docs:
        all_docs_dict[doc.id] = doc

    # Add Embedding docs, updating meta if ID exists
    for doc in embedding_docs:
        if doc.id in all_docs_dict:
            # Document already found by BM25, add embedding score to its meta
            # The original doc object from bm25_docs is in all_docs_dict
            all_docs_dict[doc.id].meta['embedding_score'] = doc.score
            all_docs_dict[doc.id].meta['retrieval_type'] = 'hybrid' # Mark as found by both
        else:
            # Document only found by embedding retriever
            all_docs_dict[doc.id] = doc
            # bm25_score would be missing or could be set to a very low value if needed for sorting

    combined_docs = list(all_docs_dict.values())

    # Sort based on scores. Higher is better.
    # This custom sort attempts to prioritize based on available scores.
    def sort_key(doc: Document):
        # Prioritize embedding score if available, then bm25 score
        # Ensure scores are float for comparison, default to -infinity if None or not found
        emb_s = doc.meta.get('embedding_score', -float('inf'))
        bm25_s = doc.meta.get('bm25_score', -float('inf'))
        
        # If a doc was retrieved by embedding, its doc.score might be the embedding score
        # If by BM25, its doc.score might be the BM25 score.
        # The meta fields 'embedding_score' and 'bm25_score' are more explicit.
        
        # Take the maximum of the explicit scores, or fallback to doc.score if only one method retrieved it.
        # If retrieval_type is 'hybrid', it means both scores should ideally be in meta.
        current_doc_score = doc.score if doc.score is not None else -float('inf')

        if doc.meta.get('retrieval_type') == 'hybrid': # Found by both
             return max(emb_s if emb_s is not None else -float('inf'), bm25_s if bm25_s is not None else -float('inf'))
        elif doc.meta.get('retrieval_type') == 'embedding':
             return emb_s if emb_s is not None else current_doc_score
        elif doc.meta.get('retrieval_type') == 'bm25':
             return bm25_s if bm25_s is not None else current_doc_score
        return current_doc_score # Fallback

    combined_docs.sort(key=sort_key, reverse=True)

    return combined_docs[:top_k_join]
# --- END OF NEWLY ADDED FUNCTION ---
