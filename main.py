import pdfplumber
import json
import os
import re
from openai import OpenAI, OpenAIError
import time

# OpenAI API Key (Set your actual API key)
client = OpenAI(api_key="YOUR-API-KEY")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = text.split("\n\n")  # Split into paragraphs
                text_data.extend([p.strip() for p in paragraphs if p.strip()])
    return text_data

def generate_questions(text, num_questions=1, retries=3):
    """
    Generate questions based on the given text using OpenAI API.

    :param text: The text content to generate questions from.
    :param num_questions: The number of questions to generate.
    :param retries: Number of retries if OpenAI API fails.
    :return: A list of generated questions.
    """
    prompt = f"""
    Given the following text, generate {num_questions} relevant questions a customer might ask:

    "{text}"

    Questions:
    """

    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            response = chat_completion.choices[0].message.content.strip()
            return re.split(r'\n\d+\. ', response) if '\n1.' in response else response.split('\n')

        except OpenAIError as e:
            print(f"⚠️ OpenAI API error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)  # Wait before retrying

    print("❌ Failed to generate questions after multiple attempts.")
    return ["Could not generate a question for this content."]  # Return fallback question

def process_pdf_and_write_jsonl(pdf_path, output_file):
    """
    Process a single PDF file, generate training data, and write to JSONL file immediately.

    :param pdf_path: Path to the PDF file.
    :param output_file: Path to the JSONL file.
    """
    pdf_text = extract_text_from_pdf(pdf_path)

    with open(output_file, "a", encoding="utf-8") as file:  # Append mode to write data instantly
        for paragraph in pdf_text:
            user_questions = generate_questions(paragraph)  # Generate queries
            
            for question in user_questions:
                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Your name is REX, a helpful AI assistant providing factual answers to user queries."
                        },
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": paragraph
                        }
                    ]
                }
                file.write(json.dumps(conversation, ensure_ascii=False) + "\n")  # Write immediately

    print(f"✅ Finished processing {os.path.basename(pdf_path)}.")

def process_all_pdfs(folder_path, output_file):
    """
    Process all PDFs in the folder one by one, writing to JSONL immediately.

    :param folder_path: Path to the folder containing PDFs.
    :param output_file: Path to the JSONL output file.
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"📄 Processing: {pdf_file}")
        process_pdf_and_write_jsonl(pdf_path, output_file)

    print(f"🎉 All PDFs processed. Data saved in {output_file}")

# Example usage
folder_path = "data"  # Folder containing PDF files
output_jsonl = "training_data.jsonl"

process_all_pdfs(folder_path, output_jsonl)
