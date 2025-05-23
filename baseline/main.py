# llm_only_baseline/main.py
import os
import sys
from getpass import getpass
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
import torch # For LLM loading

# Import from local files within llm_only_baseline
import config # from llm_only_baseline.config
from llm_processing import generate_llm_only_formatted_answer
# Removed preprocess_user_query_with_llm import as per previous discussion for a purer baseline

# Transformers and Ragas imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.utils import ComponentDevice # For device detection
from ragas import evaluate as ragas_evaluate_lib
from ragas.metrics import faithfulness, answer_relevancy

# For Colab download functionality
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def download_evaluation_data(url, local_filename):
    """Simplified download utility."""
    import requests # Moved import here to be self-contained
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            print(f"{local_filename} downloaded successfully.")
        except Exception as e: print(f"Error downloading {local_filename}: {e}")
    else:
        print(f"{local_filename} already exists.")

def load_llm_for_baseline(model_name: str, device_str: str):
    """Loads LLM and Tokenizer specifically for this baseline script."""
    print(f"\nLoading LLM for Baseline: {model_name} on device: {device_str}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_str if device_str != "cpu" else None,
            trust_remote_code=True
        )
        if device_str == "cpu" and (device_str if device_str != "cpu" else None) is None : # Check if model was actually mapped to CPU
             model.to(torch.device("cpu"))
        model.eval()
        print("LLM and Tokenizer for Baseline loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading LLM for Baseline: {e}")
        return None, None

def setup_api_keys_for_baseline():
    """Sets up API keys needed for Ragas."""
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API Key not found in environment variables (likely needed for Ragas).")
    else:
        print("OpenAI API Key found in environment variables.")
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is NOT set. Ragas LLM-based metrics might fail.")

def run_baseline_evaluation():
    """Main function for LLM-only baseline evaluation."""
    print("--- Initializing LLM-Only Baseline System ---")
    
    llm_device = ComponentDevice.resolve_device()
    llm_device_str = llm_device.to_torch_str()
    print(f"LLM will attempt to use device: {llm_device_str}")
    print(f"LLM Model for Baseline: {config.QWEN_MODEL_NAME}")

    setup_api_keys_for_baseline()

    llm_model, llm_tokenizer = load_llm_for_baseline(config.QWEN_MODEL_NAME, llm_device_str)
    if not llm_model or not llm_tokenizer:
        print("Failed to load LLM for baseline. Aborting evaluation.")
        return

    download_evaluation_data(config.EVAL_CSV_URL, config.EVAL_CSV_FILENAME)
    eval_df = None
    try:
        eval_df = pd.read_csv(config.EVAL_CSV_FILENAME)
        print(f"\nLoaded {len(eval_df)} questions for baseline evaluation from '{config.EVAL_CSV_FILENAME}'.")
    except Exception as e:
        print(f"Error loading {config.EVAL_CSV_FILENAME}: {e}. Aborting evaluation.")
        return

    baseline_eval_data_for_ragas = []
    print(f"\n--- Generating Answers with LLM-Only Baseline System (No Query Preprocessing) ---")

    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Evaluating Baseline System"):
        question = str(row['question'])
        ground_truth = str(row['ground truth'])
        
        query_to_llm = question 
        # print(f"Processing (Baseline) - Original User Query: \"{query_to_llm}\"") # Can be verbose
        
        try:
            answer_str = generate_llm_only_formatted_answer(
                user_query=query_to_llm,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer
            )
            baseline_eval_data_for_ragas.append({
                "question": question, "answer": answer_str,
                "contexts": [], "ground_truth": ground_truth
            })
        except Exception as e:
            print(f"\nError processing question (Baseline) ID {index} ('{question[:50]}...'): {e}")
            baseline_eval_data_for_ragas.append({
                "question": question, "answer": f"Error during processing: {e}",
                "contexts": [], "ground_truth": ground_truth
            })

    if not baseline_eval_data_for_ragas:
        print("No data generated for Ragas evaluation from the baseline system.")
        return

    baseline_dataset_for_ragas = Dataset.from_list(baseline_eval_data_for_ragas)
    print(f"\nBaseline Ragas evaluation dataset prepared with {len(baseline_dataset_for_ragas)} samples.")

    print("\n--- Running Ragas Evaluation for LLM-Only Baseline System ---")
    baseline_metrics_for_ragas = [faithfulness, answer_relevancy]
    print(f"Evaluating with Ragas metrics: {[m.name for m in baseline_metrics_for_ragas]}")

    ragas_results_df = None
    try:
        ragas_evaluation_object = ragas_evaluate_lib(
            dataset=baseline_dataset_for_ragas,
            metrics=baseline_metrics_for_ragas,
            raise_exceptions=False
        )
        if ragas_evaluation_object:
            ragas_results_df = ragas_evaluation_object.to_pandas()
            print("\nLLM-Only Baseline System Ragas Evaluation Results:")
            print(ragas_results_df.to_string())
        else:
            print("Ragas evaluation for baseline did not return results.")
    except Exception as e:
        print(f"Error during Ragas evaluation for baseline: {e}")

    # --- MODIFIED SECTION FOR FILE SAVING/DOWNLOADING ---
    if ragas_results_df is not None and not ragas_results_df.empty:
        output_filename = config.RAGAS_RESULTS_BASELINE_FILENAME
        
        # Always save the file to the Colab/local filesystem first
        try:
            ragas_results_df.to_csv(output_filename, index=False)
            print(f"\nBaseline Ragas evaluation results successfully saved to '{output_filename}' in the current environment's filesystem.")
            if IN_COLAB: # If in Colab, also print manual download instruction
                 print("You can find it in the Colab file browser on the left panel and download it from there.")
        except Exception as e_save:
            print(f"Error saving baseline Ragas results to CSV '{output_filename}': {e_save}")
            print("The DataFrame 'ragas_results_df' might still be available in memory if the script hasn't terminated.")
            return # Exit if save fails, as download won't work

        # Attempt automatic browser download only if IN_COLAB is True
        if IN_COLAB:
            print(f"\nAttempting to initiate browser download for '{output_filename}'...")
            try:
                from google.colab import files # Ensure it's imported in this scope
                files.download(output_filename)
                print(f"Browser download for '{output_filename}' initiated. Check your browser downloads.")
            except Exception as e_download:
                print(f"Error during automatic Colab file download initiation: {e_download}")
                print(f"This is common when running as '!python script.py'.")
                print(f"Please manually download '{output_filename}' from the Colab file browser (if the file was saved successfully above).")
    elif baseline_dataset_for_ragas: # Data prepared but Ragas eval might have failed
        print("\nRagas evaluation (Baseline) was not performed or did not produce a results DataFrame.")
        print("If data was prepared for Ragas, you could try saving it manually, e.g.:")
        print(f"# import pandas as pd")
        print(f"# pd.DataFrame(baseline_dataset_for_ragas.to_list()).to_csv('prepared_for_ragas_{output_filename}', index=False)")
    else:
        print("\nNo Ragas evaluation results DataFrame to save for the baseline system.")
    # --- END OF MODIFIED SECTION ---

    print("\n--- LLM-Only Baseline System Evaluation Complete ---")

if __name__ == "__main__":
    run_baseline_evaluation()
