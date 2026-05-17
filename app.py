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
            text = page.extract_text()
            if text:
                resume_text += text + " "

        # 1. Calculate TF-IDF Cosine Similarity Score
        documents = [resume_text, job_description]
       vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf_matrix)[0][1] * 100

     # 2. Extract Missing Keywords Logic
        keyword_vectorizer = TfidfVectorizer(stop_words='english')
        keyword_vectorizer.fit([resume_text, job_description])
        analyze_text = keyword_vectorizer.build_analyzer()
        
        job_words = set(analyze_text(job_description))
        resume_words = set(analyze_text(resume_text))

        # Find terms that exist in the job description but NOT in the resume
        missing = [word for word in job_words if word not in resume_words and len(word) > 2]

        # Format words to look clean on the frontend layout (e.g., "sql", "java")
        missing_skills = [word.upper() for word in missing[:6]]

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
