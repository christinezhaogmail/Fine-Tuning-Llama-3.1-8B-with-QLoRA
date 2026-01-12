import os
import csv
from pypdf import PdfReader
from openai import OpenAI

# Initialize the OpenAI client
# 1. GET an environment variable (Raises KeyError if missing)
# Use this for mandatory variables your script CANNOT run without.
try:
    openai_api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("api_key variable not set!")

client = OpenAI(api_key=openai_api_key)

def extract_abstract_or_intro(pdf_path):
    """Extracts the first 2 pages to get the abstract/intro context."""
    reader = PdfReader(pdf_path)
    text = ""
    # Usually, the first 2 pages contain the abstract and introduction
    for page in reader.pages[:2]:
        text += page.extract_text()
    return text

def generate_qa_pairs(paper_text, paper_title):
    prompt = f"""
    Act as an academic researcher. Based on the following research text from the paper "{paper_title}", 
    generate exactly 5 high-quality Question-Answer pairs. 
    
    The questions should cover:
    1. Core Research Problem
    2. Methodology
    3. Key Definition/Terminology
    4. Main Finding
    5. Practical Insight
    
    Format the output strictly as a CSV-compatible list with "Question|Answer" on each line.
    
    Text:
    {paper_text}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # Using gpt-4o for speed and reasoning
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# Main Processing Loop
input_folder = "./data/pdfs"
output_file = "research_qa_dataset.csv"

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Paper Title", "Question", "Answer"])

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}...")
            path = os.path.join(input_folder, filename)
            
            try:
                content = extract_abstract_or_intro(path)
                qa_data = generate_qa_pairs(content, filename)
                
                # Parsing the response (Assuming "Question|Answer" format)
                for line in qa_data.strip().split('\n'):
                    if "|" in line:
                        q, a = line.split("|", 1)
                        writer.writerow([filename, q.strip(), a.strip()])
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"Done! Results saved to {output_file}")