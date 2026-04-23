from flask import Flask, render_template, request
import os
from utils import get_resume_text, clean_and_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- UPLOAD FOLDER ----------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------- BERT (LAZY LOADING) ----------------
bert_model = None

def get_bert():
    global bert_model
    if bert_model is None:
        from sentence_transformers import SentenceTransformer
        # lighter + faster model for Render
        bert_model = SentenceTransformer('all-MiniLM-L3-v2')
    return bert_model


# ---------------- WARMUP AFTER START ----------------
@app.before_request
def warmup():
    # runs once per request cycle safely
    pass


# ---------------- SECTION EXTRACTION ----------------
def extract_sections(text):
    text = text.lower()

    sections = {
        "skills": "",
        "experience": "",
        "projects": ""
    }

    if "skills" in text:
        sections["skills"] = text.split("skills")[-1][:300]

    if "experience" in text:
        sections["experience"] = text.split("experience")[-1][:300]

    if "projects" in text:
        sections["projects"] = text.split("projects")[-1][:300]

    return sections


# ---------------- FEEDBACK ----------------
def generate_human_feedback(score, matched, missing):

    if score >= 75:
        level = "a strong fit"
    elif score >= 50:
        level = "a moderate fit"
    else:
        level = "a weak fit"

    feedback = f"This candidate appears to be {level} for the role. "

    if matched:
        feedback += "Relevant skills: " + ", ".join(list(matched)[:5]) + ". "

    if missing:
        feedback += "Missing skills: " + ", ".join(list(missing)[:5]) + ". "

    feedback += "Overall, improvement in missing areas can strengthen the profile."

    return feedback


# ---------------- ANALYSIS ----------------
def analyze_resume(text, job_desc):

    if not text or not job_desc:
        return 0, set(), set(), "Invalid input", {}

    clean_r = clean_and_tokenize(text)
    clean_j = clean_and_tokenize(job_desc)

    if not clean_r or not clean_j:
        return 0, set(), set(), "Empty processed text", {}

    # ---------------- TF-IDF ----------------
    tfidf_vec = TfidfVectorizer()

    try:
        matrix = tfidf_vec.fit_transform([clean_j, clean_r])
        tfidf_score = cosine_similarity(matrix[0:1], matrix[1:])[0][0]
    except:
        tfidf_score = 0

    # ---------------- BERT ----------------
    try:
        model = get_bert()
        emb_j = model.encode([clean_j])
        emb_r = model.encode([clean_r])
        bert_score = cosine_similarity(emb_j, emb_r)[0][0]
    except Exception as e:
        print("BERT failed:", e)
        bert_score = 0

    # ---------------- SKILL MATCH ----------------
    job_words = set(clean_j.split())
    resume_words = set(clean_r.split())

    matched = job_words.intersection(resume_words)
    missing = job_words - resume_words

    skill_score = len(matched) / len(job_words) if job_words else 0

    # ---------------- FINAL SCORE ----------------
    final_score = round(
        (0.4 * tfidf_score + 0.4 * bert_score + 0.2 * skill_score) * 100,
        2
    )

    sections = extract_sections(text)
    feedback = generate_human_feedback(final_score, matched, missing)

    return final_score, matched, missing, feedback, sections


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- RECRUITER ----------------
@app.route('/recruiter', methods=['GET', 'POST'])
def recruiter():

    if request.method == 'POST':

        job_desc = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        if not job_desc or not resume_files or not resume_files[0].filename:
            return "Please upload resumes and job description"

        results = []

        for file in resume_files:
            try:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

                text = get_resume_text(path)

                if not text or len(text.strip()) < 30:
                    continue

                score, matched, missing, feedback, sections = analyze_resume(text, job_desc)

                results.append({
                    'name': filename,
                    'score': score,
                    'matched': list(matched)[:10],
                    'missing': list(missing)[:10],
                    'feedback': feedback
                })

            except Exception as e:
                print("Error:", filename, e)

        results = sorted(results, key=lambda x: x['score'], reverse=True)

        return render_template('result.html', results=results)

    return render_template('recruiter.html')


# ---------------- CANDIDATE ----------------
@app.route('/candidate', methods=['GET', 'POST'])
def candidate():

    if request.method == 'POST':

        job_desc = request.form.get('job_description')
        file = request.files.get('resume')

        if not job_desc or not file:
            return "Please upload resume and job description"

        filename = file.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        text = get_resume_text(path)

        if not text or len(text.strip()) < 30:
            return "Resume too small or unreadable"

        score, matched, missing, feedback, sections = analyze_resume(text, job_desc)

        result = {
            'name': filename,
            'score': score,
            'matched': list(matched)[:10],
            'missing': list(missing)[:10],
            'feedback': feedback
        }

        return render_template('result.html', results=[result])

    return render_template('candidate.html')


# ---------------- RUN ----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))

    print("Starting app on port:", port)

    app.run(host="0.0.0.0", port=port)
