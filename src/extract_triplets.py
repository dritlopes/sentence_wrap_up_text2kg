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
corpus_name = 'meco'
threshold = '0.1'
window_size = 128 # 128, 256, 'sentence', 'none'
output_dir = '../data/output/step_outputs'
text_filepath = f'../data/processed/{corpus_name}_texts.csv'

# create output dir if non-existent
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if not os.path.isdir(f'{output_dir}/{dt_string}'): os.mkdir(f'{output_dir}/{dt_string}')

# read in texts from corpus
if os.path.isfile(text_filepath):
    stimuli_df = pd.read_csv(text_filepath)
else:
    stimuli_df = extract_texts(corpus_name)

# source: https://github.com/SapienzaNLP/relik
# Initialize the model with the current window size
relik = Relik.from_pretrained(model_name_full, device=device, top_k=10, window_size=window_size)

# TODO re-run on texts with tokenization errors: shaka, doping, thylacine, wed, monocle, beekeeping, nature
# running model incrementally
for text, keyword in zip(stimuli_df['text'].tolist(), stimuli_df['keyword'].tolist()):

    if keyword in 'shaka, doping, thylacine, wed, monocle, beekeeping, nature'.split(', '):

        # in the pre-processing of the texts (where extract_sentences is located),
        # make sure splitting the text gives the same words as in the eye-tracking data for the correct alignment
        words = text.split()

        for i in range(1, len(words)+1):

            context = " ".join(words[:i])
            relik_out: RelikOutput = relik(context)

            # print("=== Relik Output ===")
            # pp.pprint(relik_out)

            with open(f"{output_dir}/{dt_string}/output_step_{i:03d}_{model_name}_{corpus_name}_{keyword}_{threshold}_{window_size}.json", "w") as f:
              json.dump(relik_out.to_dict(), f, indent=4)
