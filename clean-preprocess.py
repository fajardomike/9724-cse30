import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# 1. Setup NLTK (downloads quietly in Colab)
nltk.download('stopwords', quiet=True)

# Load English stopwords but keep important negations (no, nor, not)
stop_words = set(stopwords.words('english')) - {"no", "nor", "not"}
stemmer = SnowballStemmer("english")

# Regular expressions to target possessive clitics in lowercase text
poss_s = re.compile(r"(?<=[a-zñ])'s\b")
poss_plural = re.compile(r"(?<=s)'(?=\s|$|[,;.)\"])")

def clean_and_process(text):
    # Step 1: Handling Casing (convert everything to lowercase)
    text = str(text).lower()
    
    # Step 2: Handling Clitics (remove 's and s' without leaving stray letters)
    text = poss_s.sub("", text)
    text = poss_plural.sub("", text)
    
    # Step 3: Tokenization (keep only letters and ñ, ignoring punctuation/numbers)
    # We also filter out stray single-letter characters here
    tokens = [w for w in re.findall(r"[a-zñ]+", text) if len(w) > 1]
    
    # Step 4: Stopword Removal and Stemming
    # This chops words to their root form (e.g., "amending" -> "amend")
    final_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    
    # Combine the final tokens back into a single clean text string
    return " ".join(final_tokens)

# 2. Load the raw dataset
# Ensure '2016-2026.tsv' is uploaded to your Colab session
print("Loading data...")
df = pd.read_csv('2016-2025.tsv', sep='\t', dtype=str, keep_default_na=False)

# Standardize column names just in case
df.columns = ["Year", "Ordinance_No", "Title"]

# 3. Apply the preprocessing pipeline to the Title column
print("Applying preprocessing steps (casing, clitics, tokenization, stemming)...")
df['Preprocessed_Text'] = df['Title'].apply(clean_and_process)

# 4. Keep ONLY the three requested columns
final_df = df[['Year', 'Ordinance_No', 'Preprocessed_Text']]

# 5. Save the simple output
output_filename = 'processed_ordinances.csv'
final_df.to_csv(output_filename, index=False)

print(f"Done! Successfully processed {len(final_df)} rows.")
print(f"Saved simple output to '{output_filename}'. You can now download it from the files pane.")

# Display the first 5 rows to verify it worked
final_df.head()