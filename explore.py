import pandas as pd
import matplotlib.pyplot as plt

# ---- Load Dataset ----
df = pd.read_csv('data/UpdatedResumeDataSet.csv')

# ---- Basic Info ----
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Total Resumes : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")
print(f"Column Names  : {df.columns.tolist()}")
print(f"Missing Values: {df.isnull().sum().tolist()}")

# ---- Categories ----
print("\n" + "=" * 50)
print("JOB CATEGORIES & COUNT")
print("=" * 50)
print(df['Category'].value_counts())

# ---- Sample Resume ----
print("\n" + "=" * 50)
print("SAMPLE RESUME TEXT (first 300 chars)")
print("=" * 50)
print(df['Resume'][0][:300])

# ---- Plot Category Distribution ----
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='barh', color='steelblue')
plt.title('Number of Resumes per Job Category', fontsize=14)
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.show()
print("\n✅ Chart saved as category_distribution.png")