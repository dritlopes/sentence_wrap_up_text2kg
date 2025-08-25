# Authors: Konstantin and Adrielli Lopes

import pandas as pd
import pprint
import json
import torch
import os
from datetime import datetime
from relik import Relik
from relik.inference.data.objects import RelikOutput
from process_corpus import extract_texts

# set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# print output nicely
pp = pprint.PrettyPrinter(indent=2, width=100, depth=5, compact=False)

# register date and time of output generation
now = datetime.now()
dt_string = now.strftime("_%Y_%m_%d_%H-%M-%S")

#######################################################################

model_name_full = "relik-ie/relik-cie-xl"
model_name = model_name_full.replace('relik-ie/', '')
corpus_name = 'onestop'
threshold = '0.1'
window_size = 128 # 128, 256, 'sentence', 'none'
output_dir = f'../data/output/step_outputs_{corpus_name}'
text_filepath = f'../data/processed/{corpus_name}_texts.csv'

# create output dir if non-existent
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if not os.path.isdir(f'{output_dir}/{dt_string}'): os.mkdir(f'{output_dir}/{dt_string}')

# read in texts from corpus
if os.path.isfile(text_filepath):
    texts_df = pd.read_csv(text_filepath)
else:
    texts_df, words_df = extract_texts(corpus_name)

# source: https://github.com/SapienzaNLP/relik
# Initialize the model with the current window size
relik = Relik.from_pretrained(model_name_full, device=device, top_k=10, window_size=window_size)

# running model incrementally

if corpus_name == 'meco':
    for text, keyword in zip(texts_df['text'].tolist(), texts_df['keyword'].tolist()):
        # in the pre-processing of the texts (where extract_sentences is located),
        # make sure splitting the text gives the same words as in the eye-tracking data for the correct alignment
        words = text.split(' ')
        for i in range(1, len(words)+1):
            context = " ".join(words[:i])
            relik_out: RelikOutput = relik(context)
            # print("=== Relik Output ===")
            # pp.pprint(relik_out)
            with open(
                    f"{output_dir}/{dt_string}/output_step_{i:03d}_{model_name}_{corpus_name}_{keyword}_{threshold}_{window_size}.json",
                    "w") as f:
                json.dump(relik_out.to_dict(), f, indent=4)

if corpus_name == 'onestop':
    for article_batch, article_id, difficulty_level, paragraph_id, text in zip(texts_df['article_batch'].tolist(),
                                                                             texts_df['article_id'].tolist(),
                                                                             texts_df['difficulty_level'].tolist(),
                                                                             texts_df['paragraph_id'].tolist(),
                                                                             texts_df['paragraph'].tolist()):
        # if (article_batch == 2 and article_id >= 6 and difficulty_level == 'Ele' and paragraph_id >= 3) or (article_batch == 2 and article_id >= 6 and difficulty_level == 'Adv' and paragraph_id >= 4) or (article_batch == 3):
        if ((article_batch == 2 and article_id in [7,8,9] and difficulty_level == 'Ele' and paragraph_id in [0,1,2])
                or (article_batch == 2 and article_id in [7,8,9] and difficulty_level == 'Adv' and paragraph_id in [0,1,2,3])):
            # make sure splitting text on space will give the same words as in eye mov data (tokenization alignment)
            words = text.split(' ')
            for i in range(1, len(words)+1):
                context = " ".join(words[:i])
                relik_out: RelikOutput = relik(context)
                # print("=== Relik Output ===")
                # pp.pprint(relik_out)
                filepath = (f"{output_dir}/{dt_string}/output_step_{i:03d}_{model_name}_{corpus_name}_{article_batch}-"
                            f"{article_id}-{paragraph_id}-{difficulty_level}_{threshold}_{window_size}.json")
                with open(filepath, "w") as f:
                  json.dump(relik_out.to_dict(), f, indent=4)
