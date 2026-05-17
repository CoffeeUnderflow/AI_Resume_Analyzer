import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text

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

        # 1. Calculate TF-IDF Cosine Similarity Score
        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf_matrix)[0][1] * 100

        # 2. Extract Missing Keywords Logic with a Master Tech Filter (>100 words)
        base_stop_words = text.ENGLISH_STOP_WORDS
        custom_fillers = {
            # Core Conversational & Verbs
            'looking', 'understands', 'conventions', 'huge', 'like', 'plus',
            'follows', 'experience', 'standard', 'design', 'oriented', 'object',
            'developer', 'seeking', 'requirements', 'aligned', 'stack', 'better',
            'insights', 'detail', 'highly', 'major', 'missing', 'technical',
            'knows', 'need', 'needs', 'want', 'wants', 'working', 'work', 'team',
            'skills', 'ability', 'knowledge', 'understanding', 'required', 'preferred',
            'role', 'position', 'responsibilities', 'duties', 'successful', 'candidate',
            'strong', 'excellent', 'good', 'years', 'environment', 'using', 'tools',
            
            # Action Verbs & Job Description Fluff
            'joining', 'hiring', 'build', 'maintain', 'develop', 'create', 'write',
            'implement', 'collaborate', 'support', 'manage', 'lead', 'drive', 'deliver',
            'ensure', 'improve', 'optimize', 'solve', 'help', 'assist', 'execute',
            'participate', 'contribute', 'analyze', 'evaluate', 'identify', 'review',
            'provide', 'possess', 'demonstrate', 'understand', 'perform', 'responsible',
            
            # Corporate Buzzwords & Adjectives
            'passionate', 'dynamic', 'innovative', 'motivated', 'talented', 'creative',
            'proven', 'track', 'record', 'results', 'detail', 'oriented', 'driven',
            'fast', 'paced', 'high', 'quality', 'effective', 'efficient', 'proactive',
            'exceptional', 'advanced', 'proficient', 'expert', 'professional', 'hands',
            'solid', 'deep', 'broad', 'excellent', 'outstanding', 'critical', 'analytical',
            
            # Workplace & Process Structural Terms
            'company', 'business', 'organization', 'culture', 'values', 'mission',
            'growth', 'opportunity', 'career', 'people', 'users', 'customers', 'clients',
            'projects', 'products', 'services', 'systems', 'solutions', 'applications',
            'platforms', 'processes', 'methodologies', 'practices', 'standards',
            'architecture', 'infrastructure', 'lifecycle', 'development', 'production',
            'frameworks', 'technologies', 'languages', 'tools', 'utilities',
            
            # Common Descriptive Connectors
            'day', 'today', 'daily', 'weekly', 'closely', 'cross', 'functional',
            'remote', 'hybrid', 'office', 'full', 'time', 'part', 'contract',
            'degree', 'computer', 'science', 'engineering', 'field', 'related',
            'equivalent', 'minimum', 'plus', 'preferred', 'bonus', 'desirable'
        }
        all_stop_words = base_stop_words.union(custom_fillers)

        keyword_vectorizer = TfidfVectorizer(stop_words=list(all_stop_words))
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
