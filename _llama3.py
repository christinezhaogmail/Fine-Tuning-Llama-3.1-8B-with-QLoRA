import os
import fitz  # PyMuPDF
import json
import ollama
from pydantic import BaseModel
from typing import List

# --- CONFIGURATION ---
PDF_FOLDER = "./data/pdfs"  # Change this to your folder path
MODEL_NAME = "llama3.2:3b"

# Define the structure we want for our data
class QA(BaseModel):
    question: str
    answer: str

class QAList(BaseModel):
    qa_pairs: List[QA]

def extract_abstract(pdf_path):
    """Reads the first page of a PDF to get the abstract."""
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text()
        doc.close()
        return text
    except Exception as e:
        return None

def process_pdfs():
    results = {}
    
    # Get all PDF files in the folder
    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    print(f"Found {len(files)} files. Starting local AI processing...")

    for filename in files:
        print(f"-> Processing: {filename}")
        path = os.path.join(PDF_FOLDER, filename)
        
        abstract_text = extract_abstract(path)
        if not abstract_text:
            continue

        prompt = f"""
        You are a research assistant. Below is the abstract of a research paper. 
        Generate 5 question-answer pairs based ONLY on this text.
        
        ABSTRACT:
        {abstract_text}
        """

        # Call the local Ollama model
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            # Using format ensures the model outputs clean JSON
            format=QAList.model_json_schema()
        )

        # Parse and store the JSON response
        try:
            output_data = json.loads(response['message']['content'])
            results[filename] = output_data
        except Exception as e:
            print(f"Error parsing JSON for {filename}: {e}")

    # Save everything to a file
    with open("local_dataset.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nSuccess! Data saved to local_dataset.json")

if __name__ == "__main__":
    process_pdfs()