import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Resume Analyzer is Live - Powered by CoffeeUnderflow"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description')
        file = request.files.get('resume')
        
        reader = PdfReader(file)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()

        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        score = cosine_similarity(tfidf_matrix)[0][1] * 100
        return jsonify({"match_score": f"{round(score, 2)}%"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # This line fixes the Render Port error
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
