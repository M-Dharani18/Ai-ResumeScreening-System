import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download required nltk data
nltk.download('stopwords', quiet=True)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ---- Run this file directly to clean and save dataset ----
if __name__ == "__main__":
    df = pd.read_csv('data/UpdatedResumeDataSet.csv')
    print(f"Loaded {len(df)} resumes...")

    df['cleaned_resume'] = df['Resume'].apply(clean_text)

    print("\nBefore cleaning:")
    print(df['Resume'][0][:200])
    print("\nAfter cleaning:")
    print(df['cleaned_resume'][0][:200])

    df.to_csv('data/cleaned_resume.csv', index=False)
    print("\n✅ Cleaned data saved to data/cleaned_resume.csv")