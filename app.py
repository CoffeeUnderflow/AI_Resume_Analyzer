import os
import uuid
import shutil
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from fpdf import FPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["GENERATED_PDFS"] = "generated_pdfs"
app.config["TEMP_IMAGES"] = "temp_images"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["GENERATED_PDFS"], exist_ok=True)
os.makedirs(app.config["TEMP_IMAGES"], exist_ok=True)

# In-memory sessions
sessions = {}

# -------------------------
# Sample Job Database
# -------------------------
job_database = [
    {"title": "AI Engineer", "description": "Develop AI models, TensorFlow, PyTorch, Python, Deep Learning."},
    {"title": "Data Scientist", "description": "Analyze data, build predictive models, Python, Scikit-learn, SQL."},
    {"title": "Backend Developer", "description": "Build REST APIs, Flask, Django, MongoDB, Docker."},
    {"title": "Machine Learning Engineer", "description": "Deploy ML models, Python, MLOps, cloud services."},
    {"title": "Computer Vision Engineer", "description": "Work on image recognition, OpenCV, deep learning, PyTorch."},
    {"title": "NLP Engineer", "description": "Natural Language Processing, transformers, Hugging Face, Python."},
    {"title": "Junior Python Developer", "description": "Maintain Python projects, debug issues, collaborate with team."}
]

# -------------------------
# Helper Functions
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text(pdf_path):
    """Extract text with PyPDF2 first, fallback to OCR for scanned PDFs"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            return text
    except:
        pass

    # Fallback: OCR
    text = ""
    temp_folder = os.path.join(app.config["TEMP_IMAGES"], str(uuid.uuid4()))
    os.makedirs(temp_folder, exist_ok=True)
    try:
        images = convert_from_path(pdf_path, output_folder=temp_folder)
        for img in images:
            text += pytesseract.image_to_string(img)
    except Exception as e:
        print("OCR Error:", e)
    shutil.rmtree(temp_folder, ignore_errors=True)
    return text.strip()

def recommend_jobs(resume_text, top_n=5):
    """AI-based Job Recommendation using TF-IDF + Cosine Similarity"""
    job_descriptions = [job["description"] for job in job_database]
    documents = [resume_text] + job_descriptions

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)

    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_n]

    recommended = []
    for idx in ranked_indices:
        job = job_database[idx]
        recommended.append({
            "title": job["title"],
            "company": "AI Corp",
            "location": "Remote",
            "match": f"Match Score: {similarities[idx]*100:.1f}%"
        })

    return recommended

def highlight_keywords(text):
    keywords = ["Python", "Flask", "Machine Learning", "MongoDB", "SQL", "AI", "Developer"]
    for kw in keywords:
        text = text.replace(kw, f"<mark>{kw}</mark>")
    return text

def generate_pdf_report(output_path, username, jobs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"AI Resume Report for {username}", ln=True, align="C")
    pdf.ln(10)
    for idx, job in enumerate(jobs, 1):
        pdf.cell(200, 10, txt=f"{idx}. {job['title']} - {job['company']} ({job['location']})", ln=True)
        pdf.cell(200, 10, txt=f"   Match: {job['match']}", ln=True)
        pdf.ln(5)
    pdf.output(output_path)

def cleanup_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except:
        pass

# -------------------------
# Routes
# -------------------------

# Landing Page
@app.route('/')
def home():
    return render_template('landing.html')

# Upload Page
@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('resume')
        if file and allowed_file(file.filename):
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            resume_text = extract_text(filepath)
            cleanup_file(filepath)

            if not resume_text.strip():
                return render_template('upload.html', error="❌ Resume text is empty or unreadable!")

            recommended_jobs = recommend_jobs(resume_text)
            highlighted_text = highlight_keywords(resume_text)

            pdf_filename = f"{filename}_report.pdf"
            pdf_path = os.path.join(app.config["GENERATED_PDFS"], pdf_filename)
            generate_pdf_report(pdf_path, "User", recommended_jobs)

            sessions[unique_id] = {
                "resume_text": resume_text,
                "highlighted_text": highlighted_text,
                "jobs": recommended_jobs,
                "pdf_report": pdf_filename
            }

            return redirect(url_for("dashboard", session_id=unique_id))

    return render_template('upload.html')

# Dashboard Page
@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    data = sessions.get(session_id)
    if not data:
        return redirect(url_for('upload_resume'))
    return render_template(
        'dashboard.html',
        text=data["resume_text"],
        highlighted_text=data["highlighted_text"],
        jobs=data["jobs"],
        pdf_report=data["pdf_report"]
    )

# PDF Download
@app.route('/download/<filename>')
def download_pdf(filename):
    file_path = os.path.join(app.config["GENERATED_PDFS"], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    #app.run(debug=True)
    pass
