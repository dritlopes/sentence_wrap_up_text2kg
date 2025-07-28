# Authors: Konstantin and Adrielli Lopes

import sys
import csv
import pprint
import json
import torch
import os
from datetime import datetime
from relik import Relik
from relik.inference.data.objects import RelikOutput
from utils import extract_texts, safe_field_size_limit

# set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# print output nicely
pp = pprint.PrettyPrinter(indent=2, width=100, depth=5, compact=False)

# register date and time of output generation
now = datetime.now()
dt_string = now.strftime("_%Y_%m_%d_%H-%M-%S")

# limit size of csv with output
maxInt = sys.maxsize
while True:
     try:
         csv.field_size_limit(maxInt)
         # print(f"Successfully set field size limit to: {maxInt}")
         break
     except OverflowError:
         maxInt = int(maxInt / 10)
         # print(f"Reduced field size limit to: {maxInt}")
csv._original_field_size_limit = csv.field_size_limit
csv.field_size_limit = safe_field_size_limit

#######################################################################

model_name_full = "relik-ie/relik-cie-small" # relik-ie/relik-cie-xl
model_name = model_name_full.replace('relik-ie/', '')
corpus_name = 'meco'
threshold = '0.1'
window_size = 128 # 128, 256, 'sentence', 'none'
output_dir = '../data/output/step_outputs'

# create output dir if non-existent
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if not os.path.isdir(f'{output_dir}/{dt_string}'): os.mkdir(f'{output_dir}/{dt_string}')

# read in texts from corpus
stimuli_df = extract_texts(corpus_name)

# source: https://github.com/SapienzaNLP/relik
# Initialize the model with the current window size
relik = Relik.from_pretrained(model_name_full, device="cuda", top_k=10, window_size=window_size)
# relik_out: RelikOutput = relik(stimuli_df['text'].tolist())

# running model incrementally
for text, keyword in zip(stimuli_df['text'].tolist(), stimuli_df['keyword'].tolist())[:1]:

    words = text.split(' ')

    for i in range(1, len(words)+1):

      context = " ".join(words[:i])
      relik = Relik.from_pretrained(model_name_full, device="cuda", top_k=10, window_size=window_size, threshold=threshold)
      relik_out: RelikOutput = relik(context)

      print("=== Relik Output ===")
      pp.pprint(relik_out)

      with open(f"{output_dir}/{dt_string}/output_step_{i:03d}_{keyword}_{model_name}_{corpus_name}_{threshold}_{window_size}.json", "w") as f:
          json.dump(relik_out.to_dict(), f, indent=4)