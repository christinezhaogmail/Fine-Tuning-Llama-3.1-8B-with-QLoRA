import os
import json
import glob
from openai import OpenAI
import fitz  # PyMuPDF
from tqdm import tqdm
import random

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path, max_pages=5):
    """
    Extract text from PDF, focusing on first few pages to get abstract and introduction.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Extract first few pages (usually contains abstract and intro)
        num_pages = min(len(doc), max_pages)
        for page_num in range(num_pages):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def generate_qa_pairs(paper_text, paper_filename, include_edge_case=False):
    """
    Use GPT-4 to generate Q&A pairs from the paper text.
    """
    edge_case_instruction = ""
    num_questions = 5

    if include_edge_case:
        edge_case_instruction = """
IMPORTANT: Include 1 edge-case question that reflects a misunderstanding or hallucinated detail about the paper.
For this question, provide an answer that corrects the false premise or clarifies that the paper doesn't contain that information.
Example: "Q: According to the paper, what is the value of constant XYZ?" when XYZ is not in the paper.
Answer: "The paper does not specify XYZ; in fact, that detail is not discussed."
"""

    prompt = f"""You are a research assistant who reads academic papers and creates quiz questions.

Below is text extracted from an academic research paper (filename: {paper_filename}).
The text contains the abstract and initial sections of the paper.

**Read the text and generate {num_questions} question-answer pairs** that a student might ask after reading this paper.
- Ensure the questions cover the key points or findings of the paper.
- Provide detailed answers based ONLY on the information in the provided text.
- Include a mix of question types (factual, conceptual, methodological).
- Avoid ambiguous or trivial questions.
{edge_case_instruction}

Paper Text:
{paper_text[:8000]}

Now output exactly {num_questions} Q&A pairs in JSON format, as a list of objects with "question" and "answer" fields.
Return ONLY valid JSON, no other text.

Example format:
[
  {{"question": "What is the main contribution of this paper?", "answer": "The paper proposes..."}},
  {{"question": "How did the authors evaluate their approach?", "answer": "The authors conducted..."}}
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant who generates high-quality question-answer pairs from academic papers. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON if there's extra text
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        qa_pairs = json.loads(content)
        return qa_pairs

    except Exception as e:
        print(f"Error generating Q&A for {paper_filename}: {e}")
        return []

def format_for_instruction_tuning(qa_pairs):
    """
    Format Q&A pairs into instruction-tuning format with system/user/assistant tokens.
    """
    system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
    formatted_data = []

    for qa in qa_pairs:
        user_q = qa["question"]
        assistant_a = qa["answer"]
        # Compose the prompt with system, user, assistant roles
        full_prompt = f"<|system|>{system_prompt}<|user|>{user_q}<|assistant|>{assistant_a}"
        formatted_data.append({"text": full_prompt})

    return formatted_data

def main():
    # Get all PDF files
    pdf_files = sorted(glob.glob("data/pdfs/*.pdf"))

    if len(pdf_files) != 100:
        print(f"Warning: Found {len(pdf_files)} PDF files, expected 100")

    print(f"Processing {len(pdf_files)} PDF files...")

    all_formatted_data = []

    # Decide which papers will have edge cases (about 10% of papers)
    edge_case_indices = random.sample(range(len(pdf_files)), k=max(1, len(pdf_files) // 10))

    # Process each PDF
    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Generating Q&A pairs")):
        filename = os.path.basename(pdf_path)

        # Extract text from PDF
        paper_text = extract_text_from_pdf(pdf_path)

        if not paper_text or len(paper_text) < 200:
            print(f"Skipping {filename}: insufficient text extracted")
            continue

        # Generate Q&A pairs (include edge case for selected papers)
        include_edge_case = idx in edge_case_indices
        qa_pairs = generate_qa_pairs(paper_text, filename, include_edge_case)

        if not qa_pairs:
            print(f"No Q&A pairs generated for {filename}")
            continue

        # Format for instruction tuning
        formatted = format_for_instruction_tuning(qa_pairs)
        all_formatted_data.extend(formatted)

        print(f"Generated {len(qa_pairs)} Q&A pairs from {filename} (edge case: {include_edge_case})")

    # Write to JSONL file
    output_file = "synthetic_qa.jsonl"
    with open(output_file, "w") as outfile:
        for entry in all_formatted_data:
            outfile.write(json.dumps(entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(all_formatted_data)} Q&A pairs")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
