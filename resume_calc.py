import streamlit as st
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
import string

# ---- Helper Functions ----
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(words)

def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(float(similarity[0][0]) * 100, 2)

# ---- Streamlit UI ----
st.set_page_config(page_title="Resume Match Calculator", page_icon="📄")
st.title("📄 Resume Match Score Calculator")
st.markdown("""
Upload your resume and paste a job description to calculate how well they match.
""")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("🔍 Calculate Match"):
    if uploaded_resume and job_description.strip():
        with st.spinner("Processing..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            resume_clean = preprocess_text(resume_text)
            job_clean = preprocess_text(job_description)

            score = calculate_similarity(resume_clean, job_clean)
            st.success(f"✅ Match Score: {score}%")

            if score > 70:
                st.info("💪 Great match! You're a strong candidate.")
            elif score > 40:
                st.warning("⚠️ Decent match. Consider updating your resume.")
            else:
                st.error("❌ Low match. Resume needs major improvement.")
    else:
        st.warning("Please upload a resume and paste a job description.")