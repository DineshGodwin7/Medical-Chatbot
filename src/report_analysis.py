"""
import pytesseract
from PyPDF2 import PdfReader

def analyze_report(file_path):
    # Check if the file is a PDF
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)

    # Perform analysis (Example: Extract glucose levels or any specific checks)
    analysis_result = process_report_text(text)
    return analysis_result

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

def extract_text_from_image(file_path):
    return pytesseract.image_to_string(file_path)

def process_report_text(text):
    # Example: Dummy analysis
    if "glucose" in text.lower():
        return "Glucose levels are high. Consult your doctor."
    return "Report looks normal. Keep up with your health!"
"""

"""import pytesseract
from PyPDF2 import PdfReader
import json
from transformers import BertTokenizer, pipeline

# Load pre-trained model for text generation (using BertTokenizer for BERT-based models)
tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
medical_analysis_pipeline = pipeline("text-generation", model="sentence-transformers/all-MiniLM-L6-v2", tokenizer=tokenizer)

def analyze_report(file_path):
 
    # Check if the file is a PDF or an image
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)

    # Use NLP model to process and generate the analysis
    analysis_result = process_report_with_model(text)
    return analysis_result

def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text.strip()

def extract_text_from_image(file_path):

    return pytesseract.image_to_string(file_path).strip()

def process_report_with_model(text):

    # Ensure the text is not empty
    if not text:
        return {
            "review": "Unable to process the report. No text extracted.",
            "prescription": "Please consult a doctor.",
            "instructions": "Ensure the uploaded report is clear and complete."
        }

    # Prepare the prompt for the model
    prompt = (
        f"Analyze this medical report and provide: "
        f"1. A review of the report, "
        f"2. A prescription if needed, "
        f"3. General instructions for the patient.\n\n"
        f"Report Text: {text}\n\n"
        f"Output in JSON format with keys: 'review', 'prescription', 'instructions'."
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate the analysis using the NLP model
    try:
        model_output = medical_analysis_pipeline(prompt, max_new_tokens=256, truncation=True, num_return_sequences=1)[0]['generated_text']

        # Attempt to parse the generated output into a JSON format
        analysis_result = json.loads(model_output)
    except Exception as e:
        # Handle errors gracefully
        analysis_result = {
            "review": "Unable to process the report.",
            "prescription": "Please consult a doctor.",
            "instructions": "Ensure the uploaded report is clear and complete."
        }

    return analysis_result
"""

"""import pytesseract
from PyPDF2 import PdfReader
import json
from transformers import BertTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 tokenizer and model for text generation
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def analyze_report(file_path):
    # Check if the file is a PDF or an image
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)

    # Process the report and generate analysis
    analysis_result = process_report_with_model(text)
    return analysis_result

def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text.strip()

def extract_text_from_image(file_path):

    return pytesseract.image_to_string(file_path).strip()



def process_report_with_model(text):
    if not text:
        return {
            "review": "Unable to process the report. No text extracted.",
            "prescription": "Please consult a doctor.",
            "instructions": "Ensure the uploaded report is clear and complete."
        }

    # Generate analysis using T5 model
    input_text = f"Analyze the following medical report: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the output
    output = model.generate(**inputs, max_length=512)
    analysis_result = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "review": analysis_result, 
        "prescription": "Consult your doctor for detailed advice.", 
        "instructions": "Follow your doctor's instructions."
    }
"""

import pytesseract
from PyPDF2 import PdfReader
from transformers import BertTokenizer

def analyze_report(file_path):
    # Check if the file is a PDF or an image
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)
    return text

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text.strip()

def extract_text_from_image(file_path):
    return pytesseract.image_to_string(file_path).strip()


def process_large_text_with_chunking(text, tokenizer, model, max_tokens=512):
    # Split the text into smaller chunks
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    num_tokens = tokens.size(1)
    
    if num_tokens <= max_tokens:
        # If within token limit, process normally
        inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True)
        outputs = model.generate(**inputs, max_length=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Process in chunks if token count exceeds the limit
    chunks = []
    start = 0
    while start < num_tokens:
        end = min(start + max_tokens, num_tokens)
        chunk_text = tokenizer.decode(tokens[0, start:end], skip_special_tokens=True)
        inputs = tokenizer(chunk_text, return_tensors="pt", max_length=max_tokens, truncation=True)
        outputs = model.generate(**inputs, max_length=max_tokens)
        chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        start = end

    # Combine results from all chunks
    return " ".join(chunks)


   

