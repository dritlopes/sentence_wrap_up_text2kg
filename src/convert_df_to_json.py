import json
import pandas as pd
from collections import defaultdict

# Load the CSV file
df = pd.read_csv("../data/processed/onestop_texts.csv")

for article, article_rows in df.groupby(['article_title', 'article_batch', 'article_id', 'difficulty_level']):
    article_data = article_rows.to_dict(orient="records")
    with open(f"../data/processed/onestop_paragraphs_article_{article}.json", "w", encoding='utf-8') as f:
        json.dump(article_data, f, ensure_ascii=False, indent=2)

# Convert the DataFrame to a list of dictionaries
data = df.to_dict(orient="records")

# Save out to JSON file
with open("../data/processed/onestop_paragraphs.json", "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Paragraph level to Article level
article_df = defaultdict(list)
for article_info, paragraphs in df.groupby(['article_title', 'article_batch', 'article_id', 'difficulty_level']):
    article_text = ' '.join([paragraph for paragraph in paragraphs['paragraph'].tolist()])
    article_df['article_batch'].append(article_info[1])
    article_df['article_id'].append(article_info[2])
    article_df['difficulty_level'].append(article_info[3])
    article_df['article_title'].append(article_info[0])
    article_df['article'].append(article_text)
df = pd.DataFrame(article_df)

# Convert the DataFrame to a list of dictionaries
data = df.to_dict(orient="records")

# Save out to JSON file
with open("../data/processed/onestop_articles.json", "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)