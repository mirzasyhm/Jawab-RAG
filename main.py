# main.py
import os
from getpass import getpass
import pandas as pd # For loading CSV and Ragas results
from datasets import Dataset # For Ragas dataset preparation
from tqdm.auto import tqdm # For progress bar
from IPython.display import display # For better DataFrame display in interactive environments

# Check if running in Google Colab for file download functionality
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Project-specific imports
import config.config as config
from utils.data_utils import download_file, load_haystack_documents_from_jsonl
from model.query_router import route_query # Used for basic router test earlier, not directly in Ragas eval
from utils.haystack_utils import (
    initialize_document_store,
    embed_and_write_documents,
    initialize_retrievers,
    load_qwen_llm
)
from model.llm_processing import (
    preprocess_user_query_with_llm,
    generate_formatted_answer_with_llm
)
from model.qa_system import quran_qa_system

# Haystack imports
from haystack.utils import ComponentDevice

# Ragas imports (ensure these are available from your requirements.txt)
from ragas import evaluate as ragas_evaluate_lib # Aliased to avoid conflict if 'evaluate' is used elsewhere
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# --- Configuration for Ragas Evaluation ---
EVAL_CSV_URL = "https://huggingface.co/datasets/mirzasyhm/quran/resolve/main/extracted_q_and_gt.csv"
# QA_SYSTEM_FUNCTION_NAME is directly quran_qa_system
DOWNLOAD_FILENAME_RAGAS = 'ragas_evaluation_results.csv'


def setup_api_keys():
    """Checks for API keys in environment variables and prints status."""
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API Key not found as an environment variable (needed for some Ragas metrics).")
    else:
        print("OpenAI API Key found in environment variables.")
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API Key is set for this session.")
    else:
        print("Warning: OPENAI_API_KEY is NOT set. Some Ragas metrics (like faithfulness, answer_relevancy) might fail or use a default fallback if not configured with a different LLM.")


def print_qa_results(answer, retrieved_docs, debug_info):
    """Helper to print results from the QA system."""
    print("\n--- QA System Output ---")
    print(f"Debug Info: {debug_info}")
    print("\nFinal Answer:")
    print(answer)
    if retrieved_docs:
        print(f"\nRetrieved {len(retrieved_docs)} documents for this answer:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  Doc {i+1} (ID: {doc.id}, Score: {doc.score if hasattr(doc, 'score') else 'N/A'}):")
            print(f"    Content (snippet): {doc.content[:150]}...")
            print(f"    Meta: {doc.meta}")
    else:
        print("\nNo documents were retrieved or used for this answer according to the system output.")
    print("--- End of QA System Output ---")


def main_application_flow():
    """Orchestrates the main flow of the RAG application, including Ragas evaluation."""
    print("--- Initializing Application ---")
    haystack_device = ComponentDevice.resolve_device()
    print(f"Haystack components will try to use device: {haystack_device.to_torch_str()}")
    print(f"JSONL file path: {config.JSONL_FILE_PATH}")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Qwen LLM: {config.QWEN_MODEL_NAME}")

    setup_api_keys() # Check for OpenAI API key for Ragas

    print("\n--- Phase 1: Data Loading ---")
    data_url = "https://huggingface.co/datasets/mirzasyhm/quran/resolve/main/quran_data_rag.jsonl"
    download_file(data_url, config.JSONL_FILE_PATH)
    raw_documents = load_haystack_documents_from_jsonl(config.JSONL_FILE_PATH)
    if not raw_documents:
        print("No documents loaded. Exiting application.")
        return
    print(f"Successfully loaded {len(raw_documents)} raw Haystack documents.")

    print("\n--- Phase 2: Document Store and Embeddings ---")
    document_store = initialize_document_store()
    if not embed_and_write_documents(document_store,
                                     raw_documents,
                                     config.EMBEDDING_MODEL,
                                     haystack_device):
        print("Failed to embed and write documents. Exiting.")
        return

    print("\n--- Phase 3: Retriever Initialization ---")
    bm25_retriever, embedding_retriever, query_text_embedder = initialize_retrievers(
        document_store, config.EMBEDDING_MODEL, haystack_device,
        config.TOP_K_BM25, config.TOP_K_EMBEDDING
    )
    if not bm25_retriever or not embedding_retriever or not query_text_embedder:
        print("Failed to initialize all retrievers. RAG functionality might be impaired. Exiting.")
        return

    print("\n--- Phase 4: LLM Loading ---")
    qwen_model, qwen_tokenizer = load_qwen_llm(config.QWEN_MODEL_NAME)
    if not qwen_model or not qwen_tokenizer:
        print("Failed to load Qwen LLM. Query processing and answer generation will not work. Exiting.")
        return

    print("\n--- All Initializations Complete. Ready for QA System. ---")

    print("\n--- Phase 5: Testing Quran QA System (Sample Queries) ---")
    test_queries_for_qa = [
        "Tell me about al-fatihah verse 1 to 3",
        "What does the Quran say about patience?",
        "makna dari ayat kursi",
    ]
    for user_q in test_queries_for_qa:
        final_answer, final_docs, debug_info = quran_qa_system(
            original_user_query=user_q,
            document_store=document_store,
            bm25_retriever=bm25_retriever,
            embedding_retriever=embedding_retriever,
            query_text_embedder=query_text_embedder,
            qwen_model=qwen_model,
            qwen_tokenizer=qwen_tokenizer,
            join_top_k=config.JOIN_TOP_K
        )
        print_qa_results(final_answer, final_docs, debug_info)
    print("\n--- Sample QA System Testing Complete ---")

    # --- Phase 6: Ragas Evaluation ---
    print(f"\n--- Phase 6: Ragas Evaluation from External Dataset ---")
    print(f"Loading evaluation data from: {EVAL_CSV_URL}")
    eval_df = None
    try:
        eval_df = pd.read_csv(EVAL_CSV_URL)
        print(f"Successfully loaded {len(eval_df)} questions from the CSV.")
        # print("Preview of the loaded data:")
        # display(eval_df.head()) # display might not work well in non-interactive script execution
    except Exception as e:
        print(f"Error loading CSV from URL: {e}")

    ragas_eval_data_external = []
    ragas_dataset_from_external_csv = None
    result_external_df = None

    # Check if QA system components are available (already checked for qwen_model etc.)
    if eval_df is not None and \
       qwen_model and qwen_tokenizer and \
       bm25_retriever and embedding_retriever and query_text_embedder and \
       document_store and document_store.count_documents() > 0:

        print(f"\nPreparing Data for Ragas Evaluation (using external dataset and quran_qa_system)")

        for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Evaluating Questions for Ragas"):
            question = str(row['question'])
            ground_truth = str(row['ground truth'])

            try:
                # Call your QA system
                answer, retrieved_docs, _ = quran_qa_system(
                    original_user_query=question,
                    document_store=document_store,
                    bm25_retriever=bm25_retriever,
                    embedding_retriever=embedding_retriever,
                    query_text_embedder=query_text_embedder,
                    qwen_model=qwen_model,
                    qwen_tokenizer=qwen_tokenizer,
                    join_top_k=config.JOIN_TOP_K
                )
                contexts_list = [doc.content for doc in retrieved_docs if doc and hasattr(doc, 'content') and doc.content]
                ragas_eval_data_external.append({
                    "question": question, "answer": answer,
                    "contexts": contexts_list, "ground_truth": ground_truth
                })
            except Exception as e:
                print(f"\nError processing question ID {index} ('{question[:50]}...'): {e}")
                ragas_eval_data_external.append({
                    "question": question, "answer": f"Error during processing: {e}",
                    "contexts": [], "ground_truth": ground_truth
                })

        if ragas_eval_data_external:
            ragas_dataset_from_external_csv = Dataset.from_list(ragas_eval_data_external)
            print(f"\nRagas evaluation dataset from external CSV prepared with {len(ragas_dataset_from_external_csv)} samples.")

            if ragas_dataset_from_external_csv:
                print("\n--- Running Ragas Evaluation (External CSV based data) ---")
                metrics_to_evaluate = [
                    faithfulness, answer_relevancy,
                    context_precision, context_recall
                ]
                print(f"Evaluating with metrics: {[m.name for m in metrics_to_evaluate]}")

                try:
                    result_external_ragas_obj = ragas_evaluate_lib(
                        dataset=ragas_dataset_from_external_csv,
                        metrics=metrics_to_evaluate,
                        raise_exceptions=False # Set to True to debug individual metric failures
                    )
                    print("\nRagas Evaluation Results (External CSV data):")
                    if result_external_ragas_obj:
                        result_external_df = result_external_ragas_obj.to_pandas()
                        print(result_external_df.to_string()) # Print full df to console
                        # display(result_external_df) # For interactive environments
                    else:
                        print("Ragas evaluation did not return results.")
                except ImportError: # Should not happen if imports at top are fine
                    print("Ragas library not found. Please ensure it's installed.")
                except Exception as e:
                    print(f"Error during Ragas evaluation: {e}")
                    print("Ensure Ragas is configured correctly, especially if using LLM-based metrics (e.g., API keys like OPENAI_API_KEY).")
            else:
                print("No Ragas dataset prepared from external CSV, skipping Ragas evaluation.")
        else:
            print("No data generated for Ragas evaluation from the external CSV.")
    else:
        missing_components_message = "Skipping Ragas data preparation from external CSV due to: "
        if eval_df is None: missing_components_message += "Failed to load evaluation DataFrame. "
        if not (document_store and document_store.count_documents() > 0): missing_components_message += "Document store not ready or empty. "
        if not qwen_model: missing_components_message += "Qwen model not loaded. "
        # Add other checks if needed
        print(f"\n{missing_components_message}")


    # Download Ragas results
    if result_external_df is not None and not result_external_df.empty:
        # Always save the file to the Colab filesystem first
        try:
            result_external_df.to_csv(DOWNLOAD_FILENAME_RAGAS, index=False)
            print(f"\nRagas evaluation results successfully saved to '{DOWNLOAD_FILENAME_RAGAS}' in the current Colab environment's filesystem.")
            print("You can find it in the file browser on the left panel and download it from there.")

            # Attempt files.download() only if IN_COLAB is True, but be aware it might fail if run as !python script.py
            if IN_COLAB:
                print(f"\nAttempting to initiate browser download for '{DOWNLOAD_FILENAME_RAGAS}'...")
                try:
                    # This is the line that can fail if not in a true interactive cell context
                    from google.colab import files # Ensure it's imported here if not globally
                    files.download(DOWNLOAD_FILENAME_RAGAS)
                    print(f"Browser download for '{DOWNLOAD_FILENAME_RAGAS}' initiated. Check your browser downloads.")
                except Exception as e_download:
                    print(f"Error during automatic Colab file download initiation: {e_download}")
                    print(f"This is common when running as '!python script.py'.")
                    print(f"Please manually download '{DOWNLOAD_FILENAME_RAGAS}' from the Colab file browser.")
        except Exception as e_save:
            print(f"Error saving Ragas results to CSV: {e_save}")
            print("The DataFrame 'result_external_df' might still be available in memory if the script hasn't terminated.")

    elif ragas_dataset_from_external_csv is not None: # This means data was prepared but Ragas eval might have failed
        print("\nRagas evaluation was not performed or did not produce a results DataFrame.")
        print("If data was prepared for Ragas, you could try saving it manually, e.g.:")
        print("# import pandas as pd")
        print(f"# pd.DataFrame(ragas_dataset_from_external_csv.to_list()).to_csv('prepared_for_ragas_{DOWNLOAD_FILENAME_RAGAS}', index=False)")
    else:
        print("\nNo Ragas evaluation results DataFrame to download or save.")

    print("\n--- Application Flow Complete ---")

if __name__ == "__main__":
    main_application_flow()
