# Synthetic Q&A Generation Script

## Overview
This script generates a synthetic Q&A dataset from 100 academic papers in the `data/pdfs` folder using GPT-4.

## Prerequisites
1. **OpenAI API Key**: The `OPENAI_API_KEY` environment variable must be set
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Features
- Extracts text from the first 5 pages of each PDF (abstract + introduction)
- Generates 5 Q&A pairs per paper using GPT-4
- Includes edge-case questions (~10% of papers) that test handling of incorrect premises
- Formats data in instruction-tuning format with `<|system|>`, `<|user|>`, `<|assistant|>` tokens
- Outputs to `synthetic_qa.jsonl` (one JSON object per line)

## Usage

### Run the script:
```bash
python generate_synthetic_qa.py
```

### Expected Output:
- Progress bar showing processing of 100 PDFs
- Console output showing Q&A pairs generated per paper
- Final file: `synthetic_qa.jsonl` with ~500 Q&A pairs

### Output Format:
Each line in `synthetic_qa.jsonl`:
```json
{"text": "<|system|>You are a helpful academic Q&A assistant specialized in scholarly content.<|user|>What is the main contribution of this paper?<|assistant|>The paper proposes..."}
```

## Estimated Runtime
- ~5-10 minutes for 100 papers (depends on API rate limits)
- Uses GPT-4 with temperature=0.7 for diverse questions

## Notes
- If a PDF has insufficient text, it will be skipped
- API errors are logged but don't stop the entire process
- Edge-case questions correct false premises (e.g., "The paper doesn't discuss XYZ")
