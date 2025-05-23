# qa_system.py
import config.config as config # For JOIN_TOP_K

# Import functions from other project modules
from model.llm_processing import preprocess_user_query_with_llm, generate_formatted_answer_with_llm
from model.query_router import route_query
from utils.haystack_utils import get_hybrid_retrieved_documents_hs2 # Ensure this is correctly imported

# Haystack specific imports if needed directly (e.g. DocumentStore for type hinting)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder

# The quran_qa_system function (modified to accept components)
def quran_qa_system(
    original_user_query: str,
    document_store: InMemoryDocumentStore,
    bm25_retriever: InMemoryBM25Retriever,
    embedding_retriever: InMemoryEmbeddingRetriever,
    query_text_embedder: SentenceTransformersTextEmbedder,
    qwen_model, # Actual model object
    qwen_tokenizer, # Actual tokenizer object
    join_top_k: int
):
    """
    Main QA function with initial LLM query preprocessing, routing, and formatted LLM answers.
    Implements OPTION 1: The enhanced query is used DIRECTLY for RAG retrieval.
    """
    # --- 1. Preprocess Original Query with LLM ---
    print(f"\nOriginal User Query for QA System: \"{original_user_query}\"")
    # Note: The preprocess_user_query_with_llm function itself prints logs
    new_enhanced_query = preprocess_user_query_with_llm(
        original_query=original_user_query,
        llm_model=qwen_model,
        llm_tokenizer=qwen_tokenizer
    )
    print(f"Step 1 (QA System) - Enhanced Query by LLM: \"{new_enhanced_query}\"")

    # --- 2. Route the "New Enhanced Query" ---
    mode, extracted_info = route_query(new_enhanced_query)

    # --- Path A: Modified for "exact_multiple" ---
    if mode == "exact_multiple":
        references = extracted_info.get("references", [])
        print(f"Step 2 (QA System) - Routing: EXACT MULTIPLE/RANGE LOOKUP for {len(references)} reference(s).")

        all_matched_docs = []
        if not references:
            # This case should be rare if route_query returns "exact_multiple" only when refs are found.
            return "No valid verse references were parsed from the query for exact lookup.", [], {"preprocessed_query": new_enhanced_query, "route": "exact_multiple_no_refs_parsed"}

        for ref_idx, ref in enumerate(references):
            surah_no = ref["surah_no"]
            ayah_start = ref["ayah_start"]
            ayah_end = ref["ayah_end"]
            original_match_text = ref["original_match"]

            print(f"  Processing ref {ref_idx+1}: Surah {surah_no}, Ayahs {ayah_start}-{ayah_end} (from '{original_match_text}')")

            current_ref_filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.surah_no", "operator": "==", "value": surah_no},
                    {"field": "meta.ayah_no", "operator": ">=", "value": ayah_start},
                    {"field": "meta.ayah_no", "operator": "<=", "value": ayah_end},
                ]
            }
            try:
                docs_for_this_ref = document_store.filter_documents(filters=current_ref_filters)
                if docs_for_this_ref:
                    docs_for_this_ref.sort(key=lambda d: d.meta.get('ayah_no', 0))
                    all_matched_docs.extend(docs_for_this_ref)
                    print(f"    Found {len(docs_for_this_ref)} verse(s) for this reference.")
                else:
                    print(f"    No verses found for Surah {surah_no}, Ayahs {ayah_start}-{ayah_end}.")
            except Exception as e:
                print(f"    Error filtering documents for ref {ref_idx+1}: {e}")
                continue

        if all_matched_docs:
            final_unique_docs_dict = {doc.id: doc for doc in all_matched_docs}
            final_unique_docs_list = sorted(
                list(final_unique_docs_dict.values()),
                key=lambda d: (d.meta.get('surah_no', 0), d.meta.get('ayah_no', 0))
            )
            print(f"Step 3a (QA System) - Retrieved {len(final_unique_docs_list)} unique verse(s) in total for exact lookup.")

            final_answer, docs_for_output = generate_formatted_answer_with_llm(
                original_user_query=original_user_query, # Pass original for final framing context by LLM
                retrieved_documents=final_unique_docs_list,
                llm_model=qwen_model,
                llm_tokenizer=qwen_tokenizer
            )
            return final_answer, docs_for_output, {"preprocessed_query": new_enhanced_query, "route": "exact_multiple_llm_formatted", "parsed_references": references}
        else:
            no_verse_message = f"No verses found matching the specified references in the query: '{new_enhanced_query}'."
            return no_verse_message, [], {"preprocessed_query": new_enhanced_query, "route": "exact_multiple_not_found", "parsed_references": references}

    # --- Path B: RAG based on Enhanced Query ---
    else: # mode == "rag"
        print(f"Step 2 (QA System) - Routing: RAG for enhanced query: \"{new_enhanced_query}\"")
        print(f"Step 3b (QA System) - Performing RAG hybrid retrieval using the enhanced query...")

        retrieved_rag_docs = get_hybrid_retrieved_documents_hs2(
            query=new_enhanced_query,
            bm25_retriever_node=bm25_retriever,
            embedding_retriever_node=embedding_retriever,
            query_embedder_node=query_text_embedder,
            top_k_join=join_top_k
        )

        if not retrieved_rag_docs:
            print(f"RAG retrieval found no documents for enhanced query: \"{new_enhanced_query}\"")
        else:
            print(f"RAG retrieval found {len(retrieved_rag_docs)} documents for enhanced query.")

        print(f"Step 4b (QA System) - Generating final formatted RAG answer...")
        final_answer, docs_for_output = generate_formatted_answer_with_llm(
            original_user_query=original_user_query, # Pass original for final framing context
            retrieved_documents=retrieved_rag_docs,
            llm_model=qwen_model,
            llm_tokenizer=qwen_tokenizer
        )
        final_debug_info = {
            "preprocessed_query": new_enhanced_query,
            "route": "rag_llm_formatted_Option1",
            "retrieval_query_used": new_enhanced_query
        }
        return final_answer, docs_for_output, final_debug_info
