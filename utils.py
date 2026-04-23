import PyPDF2
import docx
import re


def get_resume_text(filepath):
    """Extract text from PDF, DOCX, TXT files safely"""
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
    """Lightweight text preprocessing (NO spaCy, NO NLTK)"""

    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Custom stopwords (safe for deployment)
    stop_words = {
        "the", "and", "is", "in", "to", "of", "for", "with", "on", "at", "by",
        "an", "a", "this", "that", "are", "be", "as", "it", "from", "or",
        "was", "were", "has", "have", "had", "but", "not"
    }

    # Remove stopwords + short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)
