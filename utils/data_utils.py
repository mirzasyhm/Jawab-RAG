# data_utils.py
import os
import json
import requests
from haystack.dataclasses import Document 


def download_file(url, local_filename):
    """
    Downloads a file from a URL to a local path if it doesn't already exist.
    """
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{local_filename} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {local_filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during download of {local_filename}: {e}")
    else:
        print(f"{local_filename} already exists. Skipping download.")


def load_haystack_documents_from_jsonl(file_path):
    """
    Loads documents from a JSONL file and converts them to Haystack Document objects.
    """
    haystack_documents = []
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}. Please ensure it's downloaded.")
        return haystack_documents

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                if "page_content" not in record or "metadata" not in record:
                    print(f"Skipping line {i+1} due to missing 'page_content' or 'metadata': {line.strip()[:100]}...")
                    continue
                doc = Document(
                    id=record.get("id", f"doc_{i}"),
                    content=record["page_content"],
                    meta=record["metadata"]
                )
                haystack_documents.append(doc)
            except json.JSONDecodeError:
                print(f"Skipping line {i+1} due to JSON decode error: {line.strip()[:100]}...")
            except Exception as e:
                print(f"Error processing line {i+1}: {e} - Line: {line.strip()[:100]}...")
    print(f"Loaded {len(haystack_documents)} Haystack documents from {file_path}")
    return haystack_documents
