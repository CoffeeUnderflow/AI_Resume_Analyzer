import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description', '')
        file = request.files.get('resume')

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        reader = PdfReader(file)
        resume_text = ""
        for page in reader.pages:
            text_content = page.extract_text()
            if text_content:
                resume_text += text_content + " "

        # 1. Calculate TF-IDF Cosine Similarity Score (Keeps your overall percentage accurate)
        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf_matrix)[0][1] * 100

        # 2. New Strict Technical Match Engine Logic
        # This acts as an approved master directory of skills to scan for
        tech_keywords = [
            "JAVA", "PYTHON", "FLASK", "GIT", "PASCALCASE", 
            "CLOUD DEPLOYMENT", "VERSION CONTROL", "OBJECT-ORIENTED DESIGN",
            "KUBERNETES", "GOLANG", "DOCKER", "AWS", "AMAZON", "SQL"
        ]
        
        # Standardize casings to prevent case-sensitivity mismatches
        clean_job = job_description.upper()
        clean_resume = resume_text.upper()
        
        # Explicitly verify presence vs absence
        missing = []
        for skill in tech_keywords:
            if skill in clean_job and skill not in clean_resume:
                missing.append(skill)

        # Grab the top 6 identified terms for the frontend layout box
        missing_skills = missing[:6]

        # 3. Return keys to the frontend UI pipeline
        return jsonify({
            "match_score": f"{round(score, 2)}%",
            "missing_skills": missing_skills
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # This line fixes the Render Port error
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
