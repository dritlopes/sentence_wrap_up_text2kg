from aiohttp.log import client_logger
from openai import OpenAI
import json
import pandas as pd
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import re
import time

# take environment variables from .env, which contains OPENAI_API_KEY authorisation.
load_dotenv()

# Define constants
CORPUS = 'meco' # options: 'onestop', 'meco', 'provo'
MODEL = "gpt-4o-mini"
INPUT_FILEPATH = f"../data/processed/{CORPUS}_texts.csv"
OUTPUT_FILEPATH = f"../data/output/{MODEL}/{CORPUS}/{MODEL}_triplets_{CORPUS}.json"
N_RUNS = 1
LEVEL = ''  # options: 'article', 'paragraph' (only for onestop)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load texts from file and a prompt for each text
data = pd.read_csv(INPUT_FILEPATH)

# Define the task instruction for the model
prompt_intro = (
f"""
Task: extract all triplets from the text below, based on the information given in the text. 
A triplet is a structured representation of the form (Entity 1, Relation, Entity 2) that captures the relationship between two entities.
For each triplet, provide the following information:
- Entity: an object, event, person, location, or concept that is mentioned in the text.
- Relation: a semantic association between two entities.
For each entity or relation, provide the following information:
- Label: the unique name of the entity or relation. Avoid vague names, such as "it".
- Mention: the exact text span in the input text that refers to an entity or relation.
"""
)
# - Span_char_indices: the start and end character indices (inclusive) of the mention in the original input text, where the first character is index 0.
# - Span_word_indices: the start and end word indices (inclusive) of the mention in the original input text, where words are defined as sequences of characters separated by whitespace, and the first word is index 0. Use the exact input text provided below for calculating indices.

# Define response schema
class TripletItem(BaseModel):
    label: str
    mention: str
    # span_char_indices: list[int]
    # span_word_indices: list[int]

    def to_dict(self):
        return {
            "label": self.label,
            "mention": self.mention,
            # "span_char_indices": self.span_char_indices,
            # "span_word_indices": self.span_word_indices
        }

class Triplet(BaseModel):
    entity_1: TripletItem
    relation: TripletItem
    entity_2: TripletItem

    def to_dict(self):
        return {
            "entity_1": self.entity_1.to_dict(),
            "relation": self.relation.to_dict(),
            "entity_2": self.entity_2.to_dict()
        }

class TextResponse(BaseModel):
    triplets: list[Triplet]

    def to_dict(self):
        return {"triplets": [triplet.to_dict() for triplet in self.triplets]}

# words_df = pd.read_csv(f'../data/processed/{CORPUS}_words.csv')

for run in range(1, N_RUNS+1):

    # store responses
    responses = []

    # Iterate over texts and generate responses

    print(f"\nRun {run}...")

    counter = 0
    OUTPUT_FILEPATH_run = OUTPUT_FILEPATH.replace('.json', f'_{run}.json')

    if os.path.exists(OUTPUT_FILEPATH_run):
        with open(OUTPUT_FILEPATH_run, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        counter = len(responses)

    start_time = time.time()

    for i, row in enumerate(data.itertuples()):

        if i >= counter:

            print(f"\nProcessing text {i+1}/{len(data)}...")

            if CORPUS == 'onestop':

                if LEVEL == 'article':
                    metadata = {"article_batch": row.article_batch,
                                "article_id": row.article_id,
                                "article_title": row.article_title,
                                "difficulty_level": row.difficulty_level}

                    # split sentences in . or ." or ? or ! or ?"
                    sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s', row.article)
                    sentences = [sentence for sentence in sentences if sentence != '']

                else:
                    metadata = {"article_batch": row.article_batch,
                                 "article_id": row.article_id,
                                 "article_title": row.article_title,
                                 "difficulty_level": row.difficulty_level,
                                 "paragraph_id": row.paragraph_id,
                                 "paragraph": row.paragraph}

                    # split sentences in . or ." or ? or ! or ?"
                    sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s', row.paragraph)
                    sentences = [sentence for sentence in sentences if sentence != '']
                    # words = [word for word in row.paragraph.split(' ')]

            elif CORPUS in ['meco','provo']:

                metadata = {"text_id": row.text_id}

                # split sentences in . or ." or ? or ! or ?"
                sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s|;\s', row.text)
                sentences = [sentence for sentence in sentences if sentence != '']

                # eye_df_filtered = words_df[words_df['text_id']==row.text_id]
                # for i, sentence in enumerate(sentences):
                #     words = [word for word in sentence.split(' ') if word != '']
                #     word_ids = [word_id for word_id, word in enumerate(words)]
                #     rows_sentence = eye_df_filtered[eye_df_filtered['sent_id']==i]
                #     words2 = rows_sentence['ia'].tolist()
                #     word_ids2 = [word_id for word_id, word in enumerate(rows_sentence['ia'].tolist())]
                #     assert word_ids == word_ids2, print(row.text_id, i, '\n', words, '\n', word_ids, '\n', words2, '\n', word_ids2)

            else:
                raise NotImplementedError(f"Corpus {CORPUS} not implemented")

            print(f"Metadata: {metadata}")
            # print(f"Prompt: {prompt}")

            extracted_info = []
            input_tokens_count = 0
            output_tokens_count = 0

            for sentence_pos, sentence in enumerate(sentences):
                # sliding window of two sentences when level is article
                # if LEVEL == 'article':
                #         if sentence_pos < len(sentences) - 1:
                #             context = '. '.join(sentences[sentence_pos:sentence_pos+2])
                #         else:
                #             continue
                # else:
                context = '. '.join(sentences[:sentence_pos+1])
                prompt = prompt_intro + f"\nText: {context}\n"
                # print(f"{prompt}")
                # print("\nSending request to OpenAI...")

                response = client.responses.parse(model=MODEL,
                                                   instructions=f'You are a helpful assistant that extracts relationships between entities from text.',
                                                   input=prompt,
                                                   text_format=TextResponse,
                                                   temperature=0)

                # Extract the assistant's reply
                # print("\nResponse: \n", response)
                triplets = response.output_parsed.to_dict()
                extracted_info.append({"step": sentence_pos,"context": context} | triplets)
                # extracted_info.append({"step": word_pos, "context": context} | triplets)
                input_tokens_count += response.usage.input_tokens
                output_tokens_count += response.usage.output_tokens

            responses.append(metadata | {"extracted_triplets": extracted_info})
            # responses.append(metadata | triplets)

            print(f"Input tokens used: {input_tokens_count}, Output tokens used: {output_tokens_count}")

            # Save intermediate results (first text up to the current text)
            with open(OUTPUT_FILEPATH_run, "w", encoding='utf-8') as f:
                json.dump(responses, f, ensure_ascii=False, indent=2)
            counter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run done in {elapsed_time:.2f} seconds")
