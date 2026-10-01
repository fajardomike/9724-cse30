"""
Label each ordinance title with SDGs using the Aurora SDG classifier API.
Adds boolean columns "SDG 1" ... "SDG 17" (plus raw score columns).
"""

import time
import pandas as pd
import requests
from tqdm import tqdm

INPUT_CSV = "project.csv"
OUTPUT_CSV = "project_sdg_labeled.csv"
TEXT_COLUMN = "Title"

MODEL = "aurora-sdg-multi"  
THRESHOLD = 0.5              
URL = f"https://aurora-sdg.labs.vu.nl/classifier/classify/{MODEL}"
SLEEP = 0.2                  


def classify(text, retries=3):
   for attempt in range(retries):
       try:
           r = requests.post(URL, json={"text": text}, timeout=60)
           r.raise_for_status()
           return r.json()
       except Exception as e:
           if attempt == retries - 1:
               print(f"\nFailed: {text[:60]!r} -> {e}")
               return None
           time.sleep(2 * (attempt + 1))


def parse_scores(resp):
   """Return {sdg_number: probability}. Tolerant to small format differences."""
   scores = {}
   if not resp:
       return scores
   preds = resp.get("predictions", resp) if isinstance(resp, dict) else resp
   for p in preds:
       sdg = p.get("sdg", {})
       code = sdg.get("code") or sdg.get("id") or sdg.get("label") or ""
       digits = "".join(ch for ch in str(code).split(".")[0] if ch.isdigit())
       if digits:
           scores[int(digits)] = float(p.get("prediction", 0))
   return scores


def main():
   df = pd.read_csv(INPUT_CSV)

   # Check on the very first row 
   first = classify(str(df[TEXT_COLUMN].iloc[0]))
   print("Sample raw response:", first)

   rows = []
   for title in tqdm(df[TEXT_COLUMN].astype(str), desc="Classifying"):
       rows.append(parse_scores(classify(title)))
       time.sleep(SLEEP)

   for n in range(1, 18):
       scores = [r.get(n, 0.0) for r in rows]
       df[f"SDG {n} score"] = scores
       df[f"SDG {n}"] = [s >= THRESHOLD for s in scores]

   df.to_csv(OUTPUT_CSV, index=False)
   print(f"Saved {OUTPUT_CSV}")
   print(df[[f"SDG {n}" for n in range(1, 18)]].sum())


if __name__ == "__main__":
   main()
