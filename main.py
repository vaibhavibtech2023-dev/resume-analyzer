from flask import Flask, render_template, request
import os
import requests
from utils import get_resume_text, clean_and_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Hugging Face API
HF_API_KEY = os.environ.get("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# ---------------- BERT API ----------------
def get_bert_score(text1, text2):
    if not HF_API_KEY:
        return 0

    try:
        payload = {
            "inputs": {
                "source_sentence": text1,
                "sentences": [text2]
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            print("HF API error:", response.text)
            return 0

        result = response.json()
        return result[0]

    except Exception as e:
        print("BERT API failed:", e)
        return 0


# ---------------- FEEDBACK ----------------
def generate_feedback(score, matched, missing):
    if score >= 75:
        level = "a strong fit"
    elif score >= 50:
        level = "a moderate fit"
    else:
        level = "a weak fit"

    feedback = f"This candidate appears to be {level}. "

    if matched:
        feedback += "Matched: " + ", ".join(list(matched)[:5]) + ". "

    if missing:
        feedback += "Missing: " + ", ".join(list(missing)[:5]) + ". "

    return feedback


# ---------------- ANALYSIS ----------------
def analyze_resume(text, job_desc):
    if not text or not job_desc:
        return 0, set(), set(), "Invalid input", {}

    clean_r = clean_and_tokenize(text)
    clean_j = clean_and_tokenize(job_desc)

    # -------- TF-IDF --------
    tfidf_score = 0
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([clean_j, clean_r])
        tfidf_score = cosine_similarity(matrix[0:1], matrix[1:])[0][0]
    except:
        pass

    # -------- BERT API --------
    bert_score = get_bert_score(clean_j, clean_r)

    # fallback if API fails
    if bert_score == 0:
        bert_score = tfidf_score

    # -------- KEYWORD MATCH --------
    stop_words = {
        "the","and","is","in","to","of","for","with","on","at","by","an","a"
    }

    job_words = {w for w in clean_j.split() if w not in stop_words and len(w) > 2}
    resume_words = {w for w in clean_r.split() if w not in stop_words and len(w) > 2}

    matched = job_words.intersection(resume_words)
    missing = job_words - resume_words

    skill_score = len(matched) / (len(job_words) + 1)

    # -------- FINAL SCORE --------
    final_score = round(
        (0.5 * tfidf_score + 0.3 * bert_score + 0.2 * skill_score) * 100,
        2
    )

    # boost for demo
    final_score = min(final_score + 10, 100)

    feedback = generate_feedback(final_score, matched, missing)

    return final_score, matched, missing, feedback, {}


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recruiter", methods=["GET", "POST"])
def recruiter():
    if request.method == "POST":
        job_desc = request.form.get("job_description")
        resume_files = request.files.getlist("resumes")

        if not job_desc or not resume_files or not resume_files[0].filename:
            return "Upload resumes and job description"

        results = []

        for file in resume_files:
            try:
                path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(path)

                text = get_resume_text(path)

                score, matched, missing, feedback, _ = analyze_resume(text, job_desc)

                results.append({
                    "name": file.filename,
                    "score": score,
                    "matched": list(matched)[:10],
                    "missing": list(missing)[:10],
                    "feedback": feedback
                })

            except Exception as e:
                print("Error:", e)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return render_template("result.html", results=results)

    return render_template("recruiter.html")


@app.route("/candidate", methods=["GET", "POST"])
def candidate():
    if request.method == "POST":
        job_desc = request.form.get("job_description")
        file = request.files.get("resume")

        if not job_desc or not file:
            return "Upload resume and job description"

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        text = get_resume_text(path)

        score, matched, missing, feedback, _ = analyze_resume(text, job_desc)

        return render_template("result.html", results=[{
            "name": file.filename,
            "score": score,
            "matched": list(matched)[:10],
            "missing": list(missing)[:10],
            "feedback": feedback
        }])

    return render_template("candidate.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
