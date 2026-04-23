import PyPDF2
import docx
import re
import spacy

# Load NLP model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def get_resume_text(filepath):
    """Extract text from PDF, DOCX, TXT safely"""
    text = ""

    try:
        # PDF
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() or ""

        # DOCX
        elif filepath.endswith('.docx'):
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + " "

        # TXT
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

    except Exception as e:
        print("Error reading file:", e)

    return text


def clean_and_tokenize(text):
    """Clean + tokenize text safely"""

    if not text:
        return ""

    # Remove emails & links
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase + NLP
    doc = nlp(text.lower())

    tokens = [token.text for token in doc if not token.is_stop and not token.is_space]

    return " ".join(tokens)