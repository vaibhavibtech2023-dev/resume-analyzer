import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords

# Download once (safe fallback)
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ---------------- TEXT EXTRACTION ----------------
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


# ---------------- CLEANING ----------------
def clean_and_tokenize(text):
    """Lightweight text cleaning without spaCy"""

    if not text:
        return ""

    # Remove emails & links
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase
    text = text.lower()

    # Tokenize (simple split)
    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    return " ".join(words)
