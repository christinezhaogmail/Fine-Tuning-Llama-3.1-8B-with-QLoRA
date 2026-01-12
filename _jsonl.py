import json

# Configuration
INPUT_FILE = "local_dataset.json"
OUTPUT_FILE = "synthetic_qa.jsonl"
SYSTEM_PROMPT = "You are a helpful academic Q&A assistant specialized in scholarly content."

def generate_jsonl():
    data_to_write = []

    # 1. Load your local nested JSON file
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            local_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2. Iterate through each PDF entry
    # The key is the filename (e.g., "2511.10583v1.pdf")
    for pdf_name, content in local_data.items():
        # Access the list of pairs: content['qa_pairs']
        qas_list = content.get("qa_pairs", [])
        
        for qa in qas_list:
            user_q = qa["question"]
            assistant_a = qa["answer"]
            
            # 3. Compose the prompt using your specific template
            # Note: We use \n for clarity if your training pipeline supports it
            full_text = f"<|system|>{SYSTEM_PROMPT}<|user|>{user_q}<|assistant|>{assistant_a}"
            
            # 4. Append as a dictionary for JSONL
            data_to_write.append({"text": full_text})

    # 5. Write to JSONL file (one JSON object per line)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for entry in data_to_write:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Success! {len(data_to_write)} Q&A pairs written to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_jsonl()