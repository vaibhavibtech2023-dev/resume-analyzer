import PyPDF2
import docx
import re

def get_resume_text(filepath):
    text = ""
    try:
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif filepath.endswith('.docx'):
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + " "
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print("Error:", e)
    return text


def clean_and_tokenize(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    return " ".join(text.split())
