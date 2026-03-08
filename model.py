import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# ---- Load Cleaned Data ----
df = pd.read_csv('data/cleaned_resume.csv')
print(f"Total resumes loaded: {len(df)}")
print(f"Total categories: {df['Category'].nunique()}")

# ---- STEP 1: Convert text to numbers using TF-IDF ----
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category']

# ---- STEP 2: Split into train and test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# ---- STEP 3: Train the model ----
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---- STEP 4: Evaluate ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ---- STEP 5: Save model and tfidf ----
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save category list for app use
categories = sorted(df['Category'].unique().tolist())
with open('categories.pkl', 'wb') as f:
    pickle.dump(categories, f)

print("\n✅ model.pkl saved!")
print("✅ tfidf.pkl saved!")
print("✅ categories.pkl saved!")
print(f"\nCategories in model: {categories}")