import pickle
import PyPDF2
from preprocess import clean_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model, tfidf and categories
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('categories.pkl', 'rb') as f:
    categories = pickle.load(f)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def predict_category(resume_text):
    """Predict job category of a resume"""
    cleaned = clean_text(resume_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Get top 3 matching categories with probabilities
    proba = model.predict_proba(vectorized)[0]
    top3_indices = proba.argsort()[-3:][::-1]
    top3 = [(model.classes_[i], round(proba[i] * 100, 1)) for i in top3_indices]

    confidence = round(max(proba) * 100, 1)
    return prediction, confidence, top3

def match_resume_to_job(resume_text, job_description):
    """Score how well a resume matches a job description
       Uses a fresh TF-IDF fit on just these two docs for fair comparison
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)

    # Extract keywords from resume and job description
    resume_words = set(cleaned_resume.split())
    job_words = set(cleaned_job.split())

    # Count how many job keywords appear in resume
    if len(job_words) == 0:
        return 0.0

    matched_keywords = resume_words.intersection(job_words)
    
    # Keyword match ratio (how many JD words appear in resume)
    keyword_score = len(matched_keywords) / len(job_words)

    # Also compute cosine similarity using fresh vectorizer
    fresh_tfidf = TfidfVectorizer()
    try:
        vectors = fresh_tfidf.fit_transform([cleaned_resume, cleaned_job])
        cosine_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    except:
        cosine_score = 0

    # Combine both scores (weighted average)
    # keyword_score gets more weight as it's more meaningful for short JDs
    final_score = (keyword_score * 0.7) + (cosine_score * 0.3)
    match_score = round(final_score * 100, 2)

    return match_score